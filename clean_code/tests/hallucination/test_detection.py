"""
Hallucination Detection Framework
Detects and quantifies hallucinations in VLM/LLM outputs
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
import re
from collections import defaultdict

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not installed. Install with: pip install spacy")
    print("   Then download model: python -m spacy download en_core_web_sm")


@dataclass
class HallucinationMetrics:
    """Metrics for hallucination detection."""
    false_positive_rate: float  # Objects claimed but not present
    false_negative_rate: float  # Objects present but not mentioned
    fabrication_rate: float     # Completely made-up details
    contradiction_rate: float   # Contradicts ground truth
    grounding_score: float      # Percentage of claims grounded in reality
    precision: float            # TP / (TP + FP)
    recall: float               # TP / (TP + FN)
    f1_score: float            # Harmonic mean of precision and recall
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class HallucinationDetector:
    """Detect hallucinations in VLM/LLM outputs."""
    
    def __init__(self, ground_truth_csv: str):
        """
        Initialize hallucination detector.
        
        Args:
            ground_truth_csv: Path to ground truth annotations
        """
        self.ground_truth_csv = Path(ground_truth_csv)
        self.ground_truth_data = self._load_ground_truth()
        
        # Load spaCy model for NLP
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load spaCy model: {e}")
                print("   Run: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
    
    def _load_ground_truth(self) -> Dict:
        """Load ground truth annotations."""
        ground_truth = {}
        
        with open(self.ground_truth_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['image_id']
                
                # Parse objects
                objects_present = set()
                if row['objects_present'] and row['objects_present'] != 'null':
                    objects_present = set(obj.strip().lower() for obj in row['objects_present'].split(','))
                
                objects_absent = set()
                if row['objects_absent'] and row['objects_absent'] != 'null':
                    objects_absent = set(obj.strip().lower() for obj in row['objects_absent'].split(','))
                
                ground_truth[image_id] = {
                    'image_path': row['image_path'],
                    'scene_type': row['scene_type'],
                    'objects_present': objects_present,
                    'objects_absent': objects_absent,
                    'ocr_text': row.get('ocr_text', 'null'),
                    'persons_present': row.get('persons_present', 'null'),
                    'lighting_condition': row.get('lighting_condition', 'normal'),
                    'source': row.get('source', 'unknown')
                }
        
        print(f"✅ Loaded {len(ground_truth)} ground truth annotations")
        return ground_truth
    
    def extract_objects_from_text(self, text: str) -> Set[str]:
        """
        Extract object mentions from text using NLP.
        
        Args:
            text: VLM/LLM output text
        
        Returns:
            Set of extracted object names
        """
        if not self.nlp:
            # Fallback: simple keyword extraction
            return self._extract_objects_simple(text)
        
        # Use spaCy for NLP extraction
        doc = self.nlp(text.lower())
        
        objects = set()
        
        # Extract nouns
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                objects.add(token.lemma_)
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            # Skip pronouns and determiners
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                objects.add(chunk.root.lemma_)
        
        return objects
    
    def _extract_objects_simple(self, text: str) -> Set[str]:
        """Simple fallback object extraction without spaCy."""
        # Common object patterns
        objects = set()
        text = text.lower()
        
        # Split by common delimiters
        words = re.findall(r'\b[a-z]+\b', text)
        
        # Filter common articles and prepositions
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}
        
        objects = set(word for word in words if word not in stop_words and len(word) > 2)
        
        return objects
    
    def detect_hallucinations(self, image_id: str, output_text: str, 
                             component: str = 'vlm') -> Dict:
        """
        Detect hallucinations in VLM/LLM output.
        
        Args:
            image_id: Image identifier
            output_text: VLM/LLM output text
            component: Component being tested ('vlm', 'llm', 'ocr')
        
        Returns:
            Dictionary with detection results
        """
        if image_id not in self.ground_truth_data:
            raise ValueError(f"Image ID not found in ground truth: {image_id}")
        
        gt = self.ground_truth_data[image_id]
        
        # Extract objects from output
        claimed_objects = self.extract_objects_from_text(output_text)
        
        # Ground truth objects
        true_objects = gt['objects_present']
        false_objects = gt['objects_absent']
        
        # Calculate confusion matrix
        true_positives = claimed_objects & true_objects  # Correct claims
        false_positives = claimed_objects & false_objects  # Claimed but absent
        false_negatives = true_objects - claimed_objects  # Missed objects
        true_negatives = false_objects - claimed_objects  # Correctly not mentioned
        
        # Additional false positives (objects not in either set)
        all_known_objects = true_objects | false_objects
        unknown_claims = claimed_objects - all_known_objects
        
        # Calculate metrics
        tp_count = len(true_positives)
        fp_count = len(false_positives) + len(unknown_claims)
        fn_count = len(false_negatives)
        tn_count = len(true_negatives)
        
        # Rates
        false_positive_rate = fp_count / (fp_count + tn_count) if (fp_count + tn_count) > 0 else 0.0
        false_negative_rate = fn_count / (fn_count + tp_count) if (fn_count + tp_count) > 0 else 0.0
        
        # Precision, Recall, F1
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Fabrication rate (unknown claims)
        fabrication_rate = len(unknown_claims) / len(claimed_objects) if claimed_objects else 0.0
        
        # Contradiction rate (claimed but known absent)
        contradiction_rate = len(false_positives) / len(claimed_objects) if claimed_objects else 0.0
        
        # Grounding score (percentage of claims that are correct)
        grounding_score = tp_count / len(claimed_objects) if claimed_objects else 0.0
        
        return {
            'image_id': image_id,
            'component': component,
            'output_text': output_text,
            'claimed_objects': list(claimed_objects),
            'true_positives': list(true_positives),
            'false_positives': list(false_positives),
            'false_negatives': list(false_negatives),
            'unknown_claims': list(unknown_claims),
            'metrics': {
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate,
                'fabrication_rate': fabrication_rate,
                'contradiction_rate': contradiction_rate,
                'grounding_score': grounding_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            },
            'counts': {
                'true_positives': tp_count,
                'false_positives': fp_count,
                'false_negatives': fn_count,
                'true_negatives': tn_count,
                'claimed': len(claimed_objects),
                'ground_truth': len(true_objects)
            }
        }
    
    def batch_detect_hallucinations(self, test_results: List[Dict]) -> List[Dict]:
        """
        Batch process hallucination detection.
        
        Args:
            test_results: List of test results with image_id, output_text, component
        
        Returns:
            List of detection results
        """
        results = []
        
        for test in test_results:
            try:
                detection = self.detect_hallucinations(
                    test['image_id'],
                    test['output_text'],
                    test.get('component', 'vlm')
                )
                results.append(detection)
            except Exception as e:
                print(f"⚠️  Error processing {test['image_id']}: {e}")
        
        return results
    
    def calculate_aggregate_metrics(self, detection_results: List[Dict]) -> HallucinationMetrics:
        """
        Calculate aggregate metrics across all detections.
        
        Args:
            detection_results: List of detection results
        
        Returns:
            Aggregated hallucination metrics
        """
        if not detection_results:
            return HallucinationMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Aggregate metrics
        total_tp = sum(r['counts']['true_positives'] for r in detection_results)
        total_fp = sum(r['counts']['false_positives'] for r in detection_results)
        total_fn = sum(r['counts']['false_negatives'] for r in detection_results)
        total_tn = sum(r['counts']['true_negatives'] for r in detection_results)
        
        # Calculate rates
        fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
        fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0.0
        
        # Precision, Recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Average fabrication and contradiction rates
        avg_fabrication = sum(r['metrics']['fabrication_rate'] for r in detection_results) / len(detection_results)
        avg_contradiction = sum(r['metrics']['contradiction_rate'] for r in detection_results) / len(detection_results)
        avg_grounding = sum(r['metrics']['grounding_score'] for r in detection_results) / len(detection_results)
        
        return HallucinationMetrics(
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            fabrication_rate=avg_fabrication,
            contradiction_rate=avg_contradiction,
            grounding_score=avg_grounding,
            precision=precision,
            recall=recall,
            f1_score=f1
        )
    
    def analyze_by_component(self, detection_results: List[Dict]) -> Dict[str, HallucinationMetrics]:
        """
        Analyze hallucinations by component (VLM, LLM, OCR).
        
        Args:
            detection_results: List of detection results
        
        Returns:
            Dictionary mapping component to metrics
        """
        by_component = defaultdict(list)
        
        for result in detection_results:
            component = result.get('component', 'unknown')
            by_component[component].append(result)
        
        metrics_by_component = {}
        for component, results in by_component.items():
            metrics_by_component[component] = self.calculate_aggregate_metrics(results)
        
        return metrics_by_component
    
    def save_results(self, detection_results: List[Dict], output_path: str):
        """
        Save detection results to JSON.
        
        Args:
            detection_results: List of detection results
            output_path: Path to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detection_results, f, indent=2)
        
        print(f"✅ Detection results saved: {output_path}")
    
    def generate_summary_report(self, detection_results: List[Dict], output_path: str):
        """
        Generate summary report with aggregate metrics.
        
        Args:
            detection_results: List of detection results
            output_path: Path to save report
        """
        # Overall metrics
        overall_metrics = self.calculate_aggregate_metrics(detection_results)
        
        # By component
        component_metrics = self.analyze_by_component(detection_results)
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HALLUCINATION DETECTION SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nTotal images tested: {len(detection_results)}")
        
        report_lines.append("\n" + "-" * 80)
        report_lines.append("OVERALL METRICS")
        report_lines.append("-" * 80)
        report_lines.append(f"False Positive Rate: {overall_metrics.false_positive_rate:.4f} ({overall_metrics.false_positive_rate*100:.2f}%)")
        report_lines.append(f"False Negative Rate: {overall_metrics.false_negative_rate:.4f} ({overall_metrics.false_negative_rate*100:.2f}%)")
        report_lines.append(f"Fabrication Rate: {overall_metrics.fabrication_rate:.4f} ({overall_metrics.fabrication_rate*100:.2f}%)")
        report_lines.append(f"Contradiction Rate: {overall_metrics.contradiction_rate:.4f} ({overall_metrics.contradiction_rate*100:.2f}%)")
        report_lines.append(f"Grounding Score: {overall_metrics.grounding_score:.4f} ({overall_metrics.grounding_score*100:.2f}%)")
        report_lines.append(f"Precision: {overall_metrics.precision:.4f}")
        report_lines.append(f"Recall: {overall_metrics.recall:.4f}")
        report_lines.append(f"F1-Score: {overall_metrics.f1_score:.4f}")
        
        report_lines.append("\n" + "-" * 80)
        report_lines.append("METRICS BY COMPONENT")
        report_lines.append("-" * 80)
        
        for component, metrics in component_metrics.items():
            report_lines.append(f"\n{component.upper()}:")
            report_lines.append(f"  False Positive Rate: {metrics.false_positive_rate:.4f}")
            report_lines.append(f"  Fabrication Rate: {metrics.fabrication_rate:.4f}")
            report_lines.append(f"  Grounding Score: {metrics.grounding_score:.4f}")
            report_lines.append(f"  Precision: {metrics.precision:.4f}")
            report_lines.append(f"  Recall: {metrics.recall:.4f}")
            report_lines.append(f"  F1-Score: {metrics.f1_score:.4f}")
        
        report_lines.append("\n" + "=" * 80)
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Summary report saved: {output_path}")
        
        # Print to console
        print('\n'.join(report_lines))


def main():
    """Main function for testing hallucination detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect hallucinations in VLM/LLM outputs")
    parser.add_argument('--ground_truth', type=str,
                       default='e:/BVP_LAYER01 (2)/BVP_LAYER01/hallucination_testing/ground_truth_annotations.csv',
                       help='Path to ground truth CSV')
    parser.add_argument('--test_results', type=str, default=None,
                       help='Path to test results JSON (image_id, output_text, component)')
    parser.add_argument('--output_dir', type=str,
                       default='e:/BVP_LAYER01 (2)/BVP_LAYER01/hallucination_testing/results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = HallucinationDetector(args.ground_truth)
    
    if args.test_results:
        # Load test results
        with open(args.test_results, 'r', encoding='utf-8') as f:
            test_results = json.load(f)
        
        # Detect hallucinations
        detection_results = detector.batch_detect_hallucinations(test_results)
        
        # Save results
        output_path = Path(args.output_dir) / 'hallucination_detection_results.json'
        detector.save_results(detection_results, str(output_path))
        
        # Generate summary
        summary_path = Path(args.output_dir) / 'hallucination_summary_report.txt'
        detector.generate_summary_report(detection_results, str(summary_path))
    else:
        print("ℹ️  No test results provided. Detector initialized.")
        print("   Use --test_results to analyze VLM/LLM outputs")


if __name__ == "__main__":
    main()
