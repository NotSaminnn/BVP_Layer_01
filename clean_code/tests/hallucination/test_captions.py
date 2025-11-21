"""
Test Pixtral-12B Scene Description Quality with COCO Captions
Evaluates scene descriptions using NLG metrics: BLEU, ROUGE, METEOR, Semantic Similarity
"""

import os
import sys
import cv2
import json
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
from tqdm import tqdm

# Set API key
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = "n3ttDOXkxoavn3ClnGBcqaDdCr0HtKob"

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LUMENAA adapters
try:
    from core.adapters.pixtral_analysis import PixtralAnalysisAdapter
    print("✅ Successfully imported LUMENAA Pixtral modules")
except ImportError as e:
    print(f"❌ Failed to import LUMENAA modules: {e}")
    sys.exit(1)

# Import NLG evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    print("✅ NLTK metrics available (BLEU, METEOR)")
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️  NLTK not available - install with: pip install nltk")
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    print("✅ ROUGE scorer available")
    ROUGE_AVAILABLE = True
except ImportError:
    print("⚠️  ROUGE not available - install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    print("✅ Sentence-BERT available for semantic similarity")
    SBERT_AVAILABLE = True
except ImportError:
    print("⚠️  Sentence-BERT not available - install with: pip install sentence-transformers")
    SBERT_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    print("✅ BERTScore available")
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  BERTScore not available - install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False

try:
    from torchmetrics.multimodal import CLIPScore
    import torch
    print("✅ CLIPScore available")
    CLIPSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  CLIPScore not available - install with: pip install torchmetrics[multimodal]")
    CLIPSCORE_AVAILABLE = False


class PixtralCaptionTester:
    """Test Pixtral scene descriptions against COCO human captions."""
    
    def __init__(self, ground_truth_csv: str):
        """
        Initialize tester with ground truth captions.
        
        Args:
            ground_truth_csv: Path to CSV with reference_captions column
        """
        self.pixtral_adapter = PixtralAnalysisAdapter(
            api_key=os.environ.get("MISTRAL_API_KEY")
        )
        
        # Load ground truth
        self.ground_truth_df = pd.read_csv(ground_truth_csv)
        print(f"✅ Loaded {len(self.ground_truth_df)} ground truth annotations")
        
        # Check for reference_captions column
        if 'reference_captions' not in self.ground_truth_df.columns:
            raise ValueError(
                "Ground truth CSV missing 'reference_captions' column. "
                "Run add_coco_captions.py first!"
            )
        
        # Initialize NLG scorers
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        
        if SBERT_AVAILABLE:
            print("Loading sentence-transformers model...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence-BERT model loaded")
        
        if CLIPSCORE_AVAILABLE:
            print("Loading CLIP model for CLIPScore...")
            self.clip_scorer = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
            self.clip_scorer.eval()
            print("✅ CLIPScore model loaded")
    
    def get_pixtral_description(self, image_path: str, prompt_type: str = "detailed") -> str:
        """
        Get scene description from Pixtral-12B.
        
        Args:
            image_path: Path to image file
            prompt_type: Type of prompt ("detailed", "simple", "default")
        
        Returns:
            Scene description string
        """
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            return ""
        
        # Call Pixtral
        try:
            description = self.pixtral_adapter.analyze_scene(
                frame=frame,
                prompt_type=prompt_type
            )
            return description if description else ""
        except Exception as e:
            print(f"⚠️  Pixtral error: {e}")
            return ""
    
    def calculate_nlg_metrics(
        self, 
        hypothesis: str, 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate NLG metrics comparing hypothesis against multiple references.
        
        Args:
            hypothesis: Generated description (Pixtral output)
            references: List of reference descriptions (human captions)
        
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Tokenize for BLEU/METEOR
        hypothesis_tokens = hypothesis.lower().split()
        reference_tokens_list = [ref.lower().split() for ref in references]
        
        # BLEU Score (1-gram to 4-gram)
        if NLTK_AVAILABLE and reference_tokens_list and hypothesis_tokens:
            smoothing = SmoothingFunction().method1
            try:
                bleu1 = sentence_bleu(reference_tokens_list, hypothesis_tokens, 
                                     weights=(1, 0, 0, 0), smoothing_function=smoothing)
                bleu2 = sentence_bleu(reference_tokens_list, hypothesis_tokens, 
                                     weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                bleu4 = sentence_bleu(reference_tokens_list, hypothesis_tokens, 
                                     weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                metrics['bleu1'] = bleu1
                metrics['bleu2'] = bleu2
                metrics['bleu4'] = bleu4
            except Exception as e:
                metrics['bleu1'] = 0.0
                metrics['bleu2'] = 0.0
                metrics['bleu4'] = 0.0
        
        # METEOR Score (avg across all references)
        if NLTK_AVAILABLE and references and hypothesis:
            try:
                meteor_scores = []
                for ref in references:
                    score = meteor_score([ref.lower().split()], hypothesis.lower().split())
                    meteor_scores.append(score)
                metrics['meteor'] = sum(meteor_scores) / len(meteor_scores)
            except Exception as e:
                metrics['meteor'] = 0.0
        
        # ROUGE Scores (max across all references)
        if ROUGE_AVAILABLE and references and hypothesis:
            try:
                rouge1_scores = []
                rouge2_scores = []
                rougeL_scores = []
                for ref in references:
                    scores = self.rouge_scorer.score(ref, hypothesis)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                metrics['rouge1'] = max(rouge1_scores)
                metrics['rouge2'] = max(rouge2_scores)
                metrics['rougeL'] = max(rougeL_scores)
            except Exception as e:
                metrics['rouge1'] = 0.0
                metrics['rouge2'] = 0.0
                metrics['rougeL'] = 0.0
        
        # Semantic Similarity (max across all references)
        if SBERT_AVAILABLE and references and hypothesis:
            try:
                hyp_embedding = self.sbert_model.encode(hypothesis, convert_to_tensor=True)
                ref_embeddings = self.sbert_model.encode(references, convert_to_tensor=True)
                similarities = util.cos_sim(hyp_embedding, ref_embeddings)[0]
                metrics['semantic_similarity'] = float(similarities.max())
            except Exception as e:
                metrics['semantic_similarity'] = 0.0
        
        # BERTScore (Precision, Recall, F1 using BERT embeddings)
        if BERTSCORE_AVAILABLE and references and hypothesis:
            try:
                # BERTScore compares hypothesis against all references
                P, R, F1 = bert_score_fn(
                    [hypothesis] * len(references),
                    references,
                    lang='en',
                    model_type='microsoft/deberta-xlarge-mnli',
                    verbose=False,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                # Use maximum scores across all references
                metrics['bertscore_precision'] = float(P.max())
                metrics['bertscore_recall'] = float(R.max())
                metrics['bertscore_f1'] = float(F1.max())
            except Exception as e:
                metrics['bertscore_precision'] = 0.0
                metrics['bertscore_recall'] = 0.0
                metrics['bertscore_f1'] = 0.0
        
        # Length ratio (hypothesis vs avg reference length)
        if references and hypothesis:
            avg_ref_len = sum(len(ref.split()) for ref in references) / len(references)
            hyp_len = len(hypothesis.split())
            metrics['length_ratio'] = hyp_len / avg_ref_len if avg_ref_len > 0 else 0.0
        
        return metrics
    
    def calculate_clipscore(self, image_path: str, caption: str) -> float:
        """
        Calculate CLIPScore - measures how well caption matches the image.
        
        Args:
            image_path: Path to image file
            caption: Generated caption
        
        Returns:
            CLIPScore value (higher is better)
        """
        if not CLIPSCORE_AVAILABLE:
            return 0.0
        
        try:
            import torch
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Convert image to tensor (required for CLIPScore)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Calculate CLIPScore (caption must be a list)
            with torch.no_grad():
                score = self.clip_scorer(image_tensor, [caption])
            
            return float(score.detach().item())
        except Exception as e:
            print(f"⚠️  CLIPScore error: {e}")
            return 0.0
    
    def test_single_image(
        self, 
        image_id: str, 
        prompt_template: Dict
    ) -> Dict:
        """
        Test Pixtral description quality on a single image.
        
        Args:
            image_id: Image identifier (e.g., "coco_000000123456")
            prompt_template: Prompt template dictionary
        
        Returns:
            Dictionary with test results
        """
        # Get ground truth
        gt_row = self.ground_truth_df[self.ground_truth_df['image_id'] == image_id]
        if gt_row.empty:
            print(f"⚠️  Image not found in ground truth: {image_id}")
            return None
        
        gt = gt_row.iloc[0].to_dict()
        
        # Build image path
        script_dir = Path(__file__).parent
        image_path = script_dir / gt['image_path']
        
        if not image_path.exists():
            print(f"⚠️  Image file not found: {image_path}")
            return None
        
        # Get reference captions
        reference_captions = gt['reference_captions'].split(' ||| ')
        
        # Get Pixtral description (using concise prompt for better BLEU/ROUGE scores)
        start_time = time.time()
        pixtral_description = self.get_pixtral_description(str(image_path), prompt_type="concise")
        latency = time.time() - start_time
        
        # Verbose output
        print(f"\n{'='*70}")
        print(f"Image: {image_id}")
        print(f"Template: {prompt_template['name']}")
        print(f"Pixtral Output: {pixtral_description[:200]}...")
        print(f"Reference Captions ({len(reference_captions)}):")
        for i, cap in enumerate(reference_captions[:2], 1):
            print(f"  {i}. {cap}")
        
        # Calculate NLG metrics
        nlg_metrics = self.calculate_nlg_metrics(pixtral_description, reference_captions)
        
        # Calculate CLIPScore (image-caption matching)
        clipscore = self.calculate_clipscore(str(image_path), pixtral_description)
        nlg_metrics['clipscore'] = clipscore
        
        print(f"Metrics: BERTScore-F1={nlg_metrics.get('bertscore_f1', 0):.3f}, "
              f"CLIPScore={nlg_metrics.get('clipscore', 0):.3f}, "
              f"METEOR={nlg_metrics.get('meteor', 0):.3f}, "
              f"Semantic={nlg_metrics.get('semantic_similarity', 0):.3f}")
        print(f"{'='*70}")
        
        # Build result dictionary
        result = {
            'image_id': image_id,
            'template_id': prompt_template['id'],
            'template_name': prompt_template['name'],
            'pixtral_description': pixtral_description,
            'reference_captions': ' ||| '.join(reference_captions),
            'num_references': len(reference_captions),
            'latency_seconds': latency,
            **nlg_metrics
        }
        
        return result
    
    def run_caption_quality_test(
        self,
        prompt_templates_path: str,
        num_images: int = 50,
        output_dir: str = "results"
    ) -> Tuple[List[Dict], Dict]:
        """
        Run full caption quality test across multiple templates.
        
        Args:
            prompt_templates_path: Path to JSON with prompt templates
            num_images: Number of images to test
            output_dir: Directory to save results
        
        Returns:
            (all_results, summary_stats)
        """
        # Load prompt templates
        with open(prompt_templates_path, 'r') as f:
            templates_data = json.load(f)
        
        # Handle nested structure if present
        if isinstance(templates_data, dict) and 'prompt_templates' in templates_data:
            # Convert nested dict to list
            templates = []
            for template_id, template_data in templates_data['prompt_templates'].items():
                template = template_data.copy()
                template['id'] = template_id
                template['name'] = template_id.replace('_', ' ').title()
                templates.append(template)
        else:
            # Assume it's already a list
            templates = templates_data
        
        # Select test images
        valid_images = self.ground_truth_df[
            self.ground_truth_df['reference_captions'].notna() & 
            (self.ground_truth_df['reference_captions'] != '')
        ]['image_id'].tolist()
        
        test_image_ids = valid_images[:num_images]
        
        print(f"\n{'='*70}")
        print(f"PIXTRAL CAPTION QUALITY TEST: COCO Captions Evaluation")
        print(f"{'='*70}")
        print(f"")
        print(f"Testing Configuration:")
        print(f"  - Images: {len(test_image_ids)}")
        print(f"  - Prompt Templates: {len(templates)}")
        print(f"  - Total Tests: {len(test_image_ids) * len(templates)}")
        print(f"  - Metrics: BLEU, ROUGE, METEOR, Semantic Similarity")
        print(f"")
        
        all_results = []
        
        # Test each template
        for template_idx, template in enumerate(templates, 1):
            print(f"\n[{template_idx}/{len(templates)}] Testing: {template['name']}")
            print(f"  Temperature: {template.get('temperature', 0.7)}")
            
            template_results = []
            
            for img_idx, image_id in enumerate(test_image_ids, 1):
                print(f"  [{img_idx}/{len(test_image_ids)}] Testing image: {image_id}")
                
                result = self.test_single_image(image_id, template)
                if result:
                    template_results.append(result)
                    all_results.append(result)
                
                # API rate limiting
                time.sleep(0.5)
            
            # Calculate template average metrics
            if template_results:
                avg_bertscore = sum(r.get('bertscore_f1', 0) for r in template_results) / len(template_results)
                avg_clipscore = sum(r.get('clipscore', 0) for r in template_results) / len(template_results)
                avg_meteor = sum(r.get('meteor', 0) for r in template_results) / len(template_results)
                avg_semantic = sum(r.get('semantic_similarity', 0) for r in template_results) / len(template_results)
                avg_latency = sum(r['latency_seconds'] for r in template_results) / len(template_results)
                
                print(f"  Results: BERTScore-F1={avg_bertscore:.3f}, CLIPScore={avg_clipscore:.3f}, "
                      f"METEOR={avg_meteor:.3f}, Semantic={avg_semantic:.3f}, "
                      f"Latency={avg_latency:.2f}s")
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results
        results_df = pd.DataFrame(all_results)
        detailed_json = output_path / "pixtral_caption_detailed_results.json"
        results_df.to_json(detailed_json, orient='records', indent=2)
        print(f"\n✅ Detailed results saved: {detailed_json}")
        
        # Save CSV
        csv_path = output_path / "pixtral_caption_metrics.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"✅ CSV results saved: {csv_path}")
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results_df, templates)
        summary_json = output_path / "pixtral_caption_summary.json"
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary saved: {summary_json}")
        
        return all_results, summary
    
    def _calculate_summary_stats(self, results_df: pd.DataFrame, templates: List[Dict]) -> Dict:
        """Calculate summary statistics across all tests."""
        summary = {
            'total_tests': len(results_df),
            'num_templates': len(templates),
            'num_images': results_df['image_id'].nunique(),
            'overall_metrics': {
                'bertscore_precision': float(results_df['bertscore_precision'].mean()) if 'bertscore_precision' in results_df.columns else 0.0,
                'bertscore_recall': float(results_df['bertscore_recall'].mean()) if 'bertscore_recall' in results_df.columns else 0.0,
                'bertscore_f1': float(results_df['bertscore_f1'].mean()) if 'bertscore_f1' in results_df.columns else 0.0,
                'clipscore': float(results_df['clipscore'].mean()) if 'clipscore' in results_df.columns else 0.0,
                'bleu1': float(results_df['bleu1'].mean()),
                'bleu2': float(results_df['bleu2'].mean()),
                'bleu4': float(results_df['bleu4'].mean()),
                'rouge1': float(results_df['rouge1'].mean()),
                'rouge2': float(results_df['rouge2'].mean()),
                'rougeL': float(results_df['rougeL'].mean()),
                'meteor': float(results_df['meteor'].mean()),
                'semantic_similarity': float(results_df['semantic_similarity'].mean()),
                'avg_latency': float(results_df['latency_seconds'].mean())
            },
            'per_template': []
        }
        
        # Per-template statistics
        for template_id in results_df['template_id'].unique():
            template_data = results_df[results_df['template_id'] == template_id]
            template_info = next((t for t in templates if t['id'] == template_id), {})
            
            summary['per_template'].append({
                'template_id': template_id,
                'template_name': template_info.get('name', 'Unknown'),
                'num_tests': len(template_data),
                'metrics': {
                    'bertscore_f1': float(template_data['bertscore_f1'].mean()) if 'bertscore_f1' in template_data.columns else 0.0,
                    'clipscore': float(template_data['clipscore'].mean()) if 'clipscore' in template_data.columns else 0.0,
                    'bleu1': float(template_data['bleu1'].mean()),
                    'bleu2': float(template_data['bleu2'].mean()),
                    'bleu4': float(template_data['bleu4'].mean()),
                    'rouge1': float(template_data['rouge1'].mean()),
                    'rouge2': float(template_data['rouge2'].mean()),
                    'rougeL': float(template_data['rougeL'].mean()),
                    'meteor': float(template_data['meteor'].mean()),
                    'semantic_similarity': float(template_data['semantic_similarity'].mean()),
                    'avg_latency': float(template_data['latency_seconds'].mean())
                }
            })
        
        # Sort templates by BERTScore F1 (best overall metric)
        summary['per_template'].sort(
            key=lambda x: x['metrics']['bertscore_f1'], 
            reverse=True
        )
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test Pixtral caption quality against COCO human captions"
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        default='ground_truth_with_captions.csv',
        help='Path to ground truth CSV with reference_captions'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        default='prompt_templates.json',
        help='Path to prompt templates JSON'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=50,
        help='Number of images to test'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    ground_truth_path = script_dir / args.ground_truth
    prompts_path = script_dir / args.prompts
    
    # Check files exist
    if not ground_truth_path.exists():
        print(f"❌ Ground truth CSV not found: {ground_truth_path}")
        print(f"   Run add_coco_captions.py first to add reference captions!")
        return 1
    
    if not prompts_path.exists():
        print(f"❌ Prompt templates not found: {prompts_path}")
        return 1
    
    # Initialize tester
    tester = PixtralCaptionTester(str(ground_truth_path))
    
    # Run tests
    results, summary = tester.run_caption_quality_test(
        prompt_templates_path=str(prompts_path),
        num_images=args.num_images,
        output_dir=args.output_dir
    )
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"PIXTRAL CAPTION QUALITY TEST COMPLETE")
    print(f"{'='*70}")
    print(f"Overall Metrics:")
    print(f"  🎯 BERTScore F1: {summary['overall_metrics']['bertscore_f1']:.3f} (Contextual similarity)")
    print(f"  🎯 CLIPScore: {summary['overall_metrics']['clipscore']:.3f} (Image-caption matching)")
    print(f"  📊 METEOR: {summary['overall_metrics']['meteor']:.3f}")
    print(f"  📊 Semantic Similarity: {summary['overall_metrics']['semantic_similarity']:.3f}")
    print(f"  📊 BLEU-1: {summary['overall_metrics']['bleu1']:.3f}")
    print(f"  📊 ROUGE-L: {summary['overall_metrics']['rougeL']:.3f}")
    print(f"  ⏱️  Avg Latency: {summary['overall_metrics']['avg_latency']:.2f}s")
    print(f"\nTop 3 Templates (by BERTScore F1):")
    for i, template in enumerate(summary['per_template'][:3], 1):
        print(f"  {i}. {template['template_name']}: BERTScore={template['metrics']['bertscore_f1']:.3f}, "
              f"CLIPScore={template['metrics']['clipscore']:.3f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
