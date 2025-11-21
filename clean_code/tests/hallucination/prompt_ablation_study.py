"""
Prompt Engineering Ablation Study
Systematically test different prompt templates for hallucination reduction
"""

import json
import csv
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import time
from tqdm import tqdm

from test_hallucination_detection import HallucinationDetector, HallucinationMetrics


@dataclass
class AblationResult:
    """Result from ablation study."""
    template_name: str
    component: str
    image_id: str
    output_text: str
    metrics: Dict
    temperature: float
    prompt_used: str


class PromptAblationStudy:
    """Conduct prompt engineering ablation study."""
    
    def __init__(self, prompt_templates_path: str, ground_truth_csv: str):
        """
        Initialize ablation study.
        
        Args:
            prompt_templates_path: Path to prompt templates JSON
            ground_truth_csv: Path to ground truth annotations
        """
        self.prompt_templates_path = Path(prompt_templates_path)
        self.ground_truth_csv = Path(ground_truth_csv)
        
        # Load prompt templates
        with open(self.prompt_templates_path, 'r', encoding='utf-8') as f:
            self.templates_config = json.load(f)
        
        self.templates = self.templates_config['prompt_templates']
        self.config = self.templates_config['ablation_study_config']
        
        print(f"✅ Loaded {len(self.templates)} prompt templates")
        
        # Initialize hallucination detector
        self.detector = HallucinationDetector(str(ground_truth_csv))
        
        # Results storage
        self.ablation_results = []
    
    def format_prompt(self, template_name: str, component: str, 
                     scene_description: str = "", user_query: str = "") -> str:
        """
        Format prompt from template.
        
        Args:
            template_name: Name of template
            component: Component being tested ('vlm', 'llm', 'ocr')
            scene_description: Scene description for LLM
            user_query: User query for LLM
        
        Returns:
            Formatted prompt string
        """
        template = self.templates[template_name]
        
        if component == 'vlm':
            return template['vlm_scene_prompt']
        elif component == 'llm':
            prompt = template['llm_query_prompt']
            prompt = prompt.replace('{scene_description}', scene_description)
            prompt = prompt.replace('{user_query}', user_query)
            return prompt
        elif component == 'ocr':
            return template['ocr_prompt']
        else:
            raise ValueError(f"Unknown component: {component}")
    
    def simulate_vlm_output(self, image_id: str, template_name: str) -> str:
        """
        Simulate VLM output for testing (replace with actual VLM call).
        
        Args:
            image_id: Image identifier
            template_name: Prompt template name
        
        Returns:
            Simulated VLM description
        """
        # In actual implementation, this would call Pixtral-12B
        # For now, return placeholder that varies by template
        
        gt = self.detector.ground_truth_data[image_id]
        objects = list(gt['objects_present'])
        
        # Simulate different behaviors based on template
        if template_name in ['minimal', 'baseline', 'temperature_high']:
            # Add some hallucinated objects
            hallucinated = ['bicycle', 'phone', 'laptop'][:1]
            output = f"I see {', '.join(objects + hallucinated)}. "
            output += f"This appears to be a {gt['scene_type']} scene."
        
        elif template_name in ['constrained', 'grounding', 'adversarial_robust']:
            # Conservative, accurate output
            output = f"Objects visible: {', '.join(objects)}. "
            output += f"Scene type: {gt['scene_type']}."
        
        elif template_name == 'structured_json':
            # JSON format
            output = json.dumps({
                'objects': objects,
                'scene_type': gt['scene_type'],
                'confidence': 'high'
            })
        
        elif template_name == 'confidence_calibrated':
            # With confidence scores
            output = "Objects:\n"
            for obj in objects:
                output += f"- {obj} (Confidence: HIGH)\n"
        
        else:
            # Default behavior
            output = f"This image shows {', '.join(objects)}."
        
        return output
    
    def simulate_llm_output(self, scene_description: str, user_query: str, 
                           template_name: str) -> str:
        """
        Simulate LLM output for testing (replace with actual LLM call).
        
        Args:
            scene_description: VLM scene description
            user_query: User query
            template_name: Prompt template name
        
        Returns:
            Simulated LLM response
        """
        # In actual implementation, this would call Mistral-Small
        # For now, return placeholder
        
        if template_name in ['minimal', 'baseline']:
            # May add information not in scene
            output = f"Based on the scene, {user_query.lower()}. "
            output += "I also notice there might be a window nearby."
        
        elif template_name in ['constrained', 'grounding']:
            # Stick to scene description
            output = f"According to the scene description: {scene_description[:100]}..."
        
        else:
            output = f"Response to '{user_query}' based on scene."
        
        return output
    
    def run_ablation_for_template(self, template_name: str, 
                                  images_to_test: List[str],
                                  component: str = 'vlm') -> List[AblationResult]:
        """
        Run ablation study for a single template.
        
        Args:
            template_name: Name of template to test
            images_to_test: List of image IDs to test
            component: Component to test ('vlm', 'llm', 'ocr')
        
        Returns:
            List of ablation results
        """
        template = self.templates[template_name]
        temperature = template['temperature']
        results = []
        
        for image_id in tqdm(images_to_test, desc=f"Testing {template_name}", leave=False):
            # Format prompt
            prompt = self.format_prompt(template_name, component)
            
            # Simulate model output (replace with actual API call)
            if component == 'vlm':
                output = self.simulate_vlm_output(image_id, template_name)
            elif component == 'llm':
                # Need scene description first
                scene_desc = self.simulate_vlm_output(image_id, 'baseline')
                user_query = "What objects are nearby?"
                prompt = self.format_prompt(template_name, component, scene_desc, user_query)
                output = self.simulate_llm_output(scene_desc, user_query, template_name)
            else:
                output = "OCR text extraction placeholder"
            
            # Detect hallucinations
            detection = self.detector.detect_hallucinations(image_id, output, component)
            
            # Store result
            result = AblationResult(
                template_name=template_name,
                component=component,
                image_id=image_id,
                output_text=output,
                metrics=detection['metrics'],
                temperature=temperature,
                prompt_used=prompt
            )
            results.append(result)
        
        return results
    
    def run_full_ablation_study(self, num_images: int = None) -> Dict:
        """
        Run complete ablation study across all templates.
        
        Args:
            num_images: Number of images to test (None = all)
        
        Returns:
            Dictionary with ablation results
        """
        print("\n" + "=" * 80)
        print("PROMPT ENGINEERING ABLATION STUDY")
        print("=" * 80 + "\n")
        
        # Select images to test
        all_image_ids = list(self.detector.ground_truth_data.keys())
        if num_images:
            images_to_test = all_image_ids[:num_images]
        else:
            images_to_test = all_image_ids
        
        print(f"Testing {len(images_to_test)} images across {len(self.templates)} templates")
        print(f"Total tests: {len(images_to_test) * len(self.templates)}\n")
        
        # Run ablation for each template
        all_results = []
        template_metrics = {}
        
        test_order = self.config.get('test_order', list(self.templates.keys()))
        
        for template_name in tqdm(test_order, desc="Template progress"):
            # Test template
            results = self.run_ablation_for_template(
                template_name, 
                images_to_test,
                component='vlm'  # Focus on VLM for now
            )
            
            all_results.extend(results)
            
            # Calculate aggregate metrics for this template
            detection_results = [
                {
                    'image_id': r.image_id,
                    'component': r.component,
                    'output_text': r.output_text,
                    'metrics': r.metrics,
                    'counts': {
                        'true_positives': 0,  # Would be calculated from detection
                        'false_positives': 0,
                        'false_negatives': 0,
                        'true_negatives': 0
                    }
                }
                for r in results
            ]
            
            # Calculate metrics
            metrics = self._calculate_template_metrics(results)
            template_metrics[template_name] = metrics
        
        # Save results
        self._save_ablation_results(all_results, template_metrics)
        
        # Generate comparison report
        self._generate_comparison_report(template_metrics)
        
        return {
            'results': all_results,
            'template_metrics': template_metrics
        }
    
    def _calculate_template_metrics(self, results: List[AblationResult]) -> Dict:
        """Calculate aggregate metrics for a template."""
        if not results:
            return {}
        
        # Average metrics across all tests
        avg_metrics = {
            'false_positive_rate': sum(r.metrics['false_positive_rate'] for r in results) / len(results),
            'false_negative_rate': sum(r.metrics['false_negative_rate'] for r in results) / len(results),
            'fabrication_rate': sum(r.metrics['fabrication_rate'] for r in results) / len(results),
            'contradiction_rate': sum(r.metrics['contradiction_rate'] for r in results) / len(results),
            'grounding_score': sum(r.metrics['grounding_score'] for r in results) / len(results),
            'precision': sum(r.metrics['precision'] for r in results) / len(results),
            'recall': sum(r.metrics['recall'] for r in results) / len(results),
            'f1_score': sum(r.metrics['f1_score'] for r in results) / len(results),
            'temperature': results[0].temperature,
            'num_tests': len(results)
        }
        
        return avg_metrics
    
    def _save_ablation_results(self, results: List[AblationResult], 
                               template_metrics: Dict):
        """Save ablation results to files."""
        output_dir = self.ground_truth_csv.parent / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        detailed_path = output_dir / 'ablation_detailed_results.json'
        detailed_data = [
            {
                'template': r.template_name,
                'component': r.component,
                'image_id': r.image_id,
                'output_text': r.output_text,
                'metrics': r.metrics,
                'temperature': r.temperature
            }
            for r in results
        ]
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2)
        
        print(f"\n✅ Detailed results saved: {detailed_path}")
        
        # Save aggregate metrics
        metrics_path = output_dir / 'ablation_template_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(template_metrics, f, indent=2)
        
        print(f"✅ Template metrics saved: {metrics_path}")
        
        # Save CSV for easy analysis
        csv_path = output_dir / 'ablation_metrics_summary.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['template', 'false_positive_rate', 'false_negative_rate', 
                         'fabrication_rate', 'contradiction_rate', 'grounding_score', 
                         'precision', 'recall', 'f1_score', 'temperature', 'num_tests']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for template_name, metrics in template_metrics.items():
                row = {'template': template_name}
                row.update(metrics)
                writer.writerow(row)
        
        print(f"✅ CSV summary saved: {csv_path}")
    
    def _generate_comparison_report(self, template_metrics: Dict):
        """Generate comparison report across templates."""
        output_dir = self.ground_truth_csv.parent / 'results'
        report_path = output_dir / 'ablation_comparison_report.txt'
        
        lines = []
        lines.append("=" * 80)
        lines.append("PROMPT ENGINEERING ABLATION STUDY - COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"\nTotal templates tested: {len(template_metrics)}")
        
        # Sort by F1 score
        sorted_templates = sorted(template_metrics.items(), 
                                 key=lambda x: x[1]['f1_score'], 
                                 reverse=True)
        
        lines.append("\n" + "-" * 80)
        lines.append("RANKING BY F1-SCORE (Best to Worst)")
        lines.append("-" * 80)
        
        for rank, (template_name, metrics) in enumerate(sorted_templates, 1):
            lines.append(f"\n{rank}. {template_name.upper()}")
            lines.append(f"   F1-Score: {metrics['f1_score']:.4f}")
            lines.append(f"   Precision: {metrics['precision']:.4f}")
            lines.append(f"   Recall: {metrics['recall']:.4f}")
            lines.append(f"   False Positive Rate: {metrics['false_positive_rate']:.4f}")
            lines.append(f"   Fabrication Rate: {metrics['fabrication_rate']:.4f}")
            lines.append(f"   Grounding Score: {metrics['grounding_score']:.4f}")
            lines.append(f"   Temperature: {metrics['temperature']}")
        
        # Best performers by metric
        lines.append("\n" + "-" * 80)
        lines.append("BEST PERFORMERS BY METRIC")
        lines.append("-" * 80)
        
        metrics_to_rank = {
            'Lowest False Positive Rate': ('false_positive_rate', False),
            'Lowest Fabrication Rate': ('fabrication_rate', False),
            'Highest Grounding Score': ('grounding_score', True),
            'Highest Precision': ('precision', True),
            'Highest Recall': ('recall', True),
            'Highest F1-Score': ('f1_score', True)
        }
        
        for label, (metric_key, reverse) in metrics_to_rank.items():
            best = sorted(template_metrics.items(), 
                         key=lambda x: x[1][metric_key], 
                         reverse=reverse)[0]
            lines.append(f"\n{label}: {best[0]} ({best[1][metric_key]:.4f})")
        
        lines.append("\n" + "=" * 80)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Comparison report saved: {report_path}\n")
        
        # Print to console
        print('\n'.join(lines))


def main():
    """Main function to run ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run prompt engineering ablation study")
    parser.add_argument('--templates', type=str,
                       default='e:/BVP_LAYER01 (2)/BVP_LAYER01/hallucination_testing/prompt_templates.json',
                       help='Path to prompt templates JSON')
    parser.add_argument('--ground_truth', type=str,
                       default='e:/BVP_LAYER01 (2)/BVP_LAYER01/hallucination_testing/ground_truth_annotations.csv',
                       help='Path to ground truth CSV')
    parser.add_argument('--num_images', type=int, default=None,
                       help='Number of images to test (None = all)')
    
    args = parser.parse_args()
    
    # Initialize and run study
    study = PromptAblationStudy(args.templates, args.ground_truth)
    results = study.run_full_ablation_study(num_images=args.num_images)
    
    print("\n✅ Ablation study complete!")
    print(f"   Results saved in: {Path(args.ground_truth).parent / 'results'}")


if __name__ == "__main__":
    main()
