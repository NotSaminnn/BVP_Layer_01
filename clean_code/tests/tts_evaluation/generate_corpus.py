"""
Script 1: Generate LUMENAA Response Corpus

Creates 1000 diverse text responses for TTS evaluation covering:
- Scene descriptions (200)
- Object location (200)
- Navigation instructions (200)
- Document reading (150)
- Person recognition (150)
- Warnings/alerts (100)

Output: data/response_corpus.csv
"""

import csv
import json
import random
from pathlib import Path
from typing import List, Dict

# Set random seed for reproducibility
random.seed(42)

# Response templates by category
RESPONSE_TEMPLATES = {
    "scene_description": [
        "A {person} {action} {location}",
        "There is a {object} on the {surface} near a {landmark}",
        "{count} people are {action} in a {location}",
        "A {vehicle} is parked {position} a {landmark}",
        "The room contains a {furniture} and a {object}",
        "An outdoor scene with {object} and {weather}",
        "A {animal} {action} on a {surface}",
        "{person} wearing a {clothing} {action}",
        "A {room_type} with {furniture} and {lighting}",
        "A street scene with {vehicles} and {landmarks}",
    ],
    "object_location": [
        "The {object} is on the {surface} to your {direction}, about {distance} away",
        "There's a {object} {position} the {landmark}, approximately {distance} from you",
        "The {object} is located {direction} at {distance}, {height} level",
        "You'll find the {object} on your {direction} side, {distance} ahead",
        "A {object} is positioned {position} you at {height} height",
        "The {object} is {distance} to your {direction}, next to the {landmark}",
        "There is a {object} {direction} of your current position, {distance} away",
        "The {object} can be found at {position}, about {distance} from here",
    ],
    "navigation": [
        "Walk forward {count} steps, then turn {direction} at the {landmark}",
        "Turn {direction} and proceed {distance} until you reach the {destination}",
        "Move {direction} for {distance}, the {landmark} will be on your {side}",
        "Go straight for {count} steps, then turn {direction} at the {object}",
        "Take {count} steps forward, you'll feel the {surface} change to {texture}",
        "Turn {angle} degrees {direction}, then walk {distance}",
        "Follow the {landmark} on your {side} for about {distance}",
        "Stop at the {landmark}, then turn {direction} and continue {distance}",
    ],
    "document_reading": [
        "Invoice number {invoice_num}, total amount ${amount}, due date {date}",
        "Product label: {product_name}, {quantity} {unit}, expires {date}",
        "Prescription: {medication} {dosage}, take {frequency} with {instructions}",
        "Address: {street_num} {street_name}, {city}, {state} {zipcode}",
        "Phone number: {phone}, extension {extension}",
        "Email: {email_local}@{email_domain}.com",
        "Date: {month} {day}, {year}, time: {time}",
        "Receipt: Item {item}, quantity {qty}, price ${price}, subtotal ${subtotal}",
    ],
    "person_recognition": [
        "{name} is standing {position} you, wearing a {clothing}",
        "I recognize {name}, they are {distance} away, facing {direction}",
        "{name} is in the {location}, wearing {color} {clothing}",
        "That's {name}, approximately {distance} {direction}, height about {height}",
        "{name} is approaching from your {direction}, about {distance} away",
        "I see {name} near the {landmark}, wearing a {accessory}",
        "{name} is {distance} to your {direction}, carrying a {object}",
    ],
    "warning": [
        "Caution: {obstacle} detected {distance} ahead, stop immediately",
        "Warning: Step {direction} detected, proceed carefully",
        "Alert: {hazard} on your {direction} at {distance}",
        "Danger: {obstacle} directly ahead, {action} recommended",
        "Caution: Uneven surface detected, slow down",
        "Warning: {vehicle} approaching from your {direction} at {distance}",
        "Alert: {weather} ahead, proceed with caution",
        "Caution: Low clearance {distance} ahead at {height} height",
    ]
}

# Vocabulary for template filling
VOCABULARY = {
    "person": ["person", "man", "woman", "child", "cyclist", "pedestrian", "jogger"],
    "action": ["walking", "standing", "sitting", "running", "cycling", "crossing the street", "waiting"],
    "location": ["park", "street", "sidewalk", "intersection", "plaza", "parking lot", "building entrance"],
    "object": ["chair", "table", "cup", "bag", "sign", "bench", "bike", "tree", "lamp", "trash bin"],
    "surface": ["table", "desk", "counter", "shelf", "floor", "ground", "bench"],
    "landmark": ["door", "window", "wall", "pillar", "tree", "building", "sign"],
    "count": ["two", "three", "four", "five", "several"],
    "vehicle": ["car", "bus", "truck", "bicycle", "motorcycle", "van"],
    "position": ["beside", "near", "in front of", "behind", "next to", "across from"],
    "furniture": ["sofa", "chair", "table", "desk", "bed", "cabinet"],
    "weather": ["clear skies", "cloudy weather", "light rain", "sunny conditions"],
    "animal": ["dog", "cat", "bird", "squirrel"],
    "clothing": ["jacket", "shirt", "dress", "coat", "hat"],
    "room_type": ["living room", "bedroom", "kitchen", "office", "hallway"],
    "lighting": ["natural light", "bright lighting", "dim lighting", "lamp illumination"],
    "vehicles": ["cars", "buses", "bicycles", "motorcycles"],
    "landmarks": ["buildings", "trees", "signs", "traffic lights"],
    "direction": ["left", "right", "straight ahead", "forward", "backward"],
    "distance": ["1 foot", "2 feet", "3 feet", "5 feet", "10 feet", "1 meter", "2 meters"],
    "height": ["waist", "chest", "head", "floor", "eye"],
    "side": ["left", "right"],
    "destination": ["doorway", "intersection", "bench", "entrance", "exit"],
    "texture": ["carpet", "tile", "concrete", "wood", "grass"],
    "angle": ["45", "90", "180"],
    "invoice_num": ["A12345", "INV-2024-001", "5678", "BV-3421"],
    "amount": ["150.00", "89.99", "250.50", "1200.00"],
    "date": ["November 30th, 2025", "December 15th, 2025", "January 1st, 2026"],
    "product_name": ["Milk", "Orange Juice", "Cereal", "Bread", "Coffee"],
    "quantity": ["1", "2", "500", "12"],
    "unit": ["liter", "ounces", "grams", "pieces"],
    "medication": ["Aspirin", "Ibuprofen", "Amoxicillin", "Vitamin D"],
    "dosage": ["500mg", "200mg", "100mg", "1000 IU"],
    "frequency": ["twice daily", "once daily", "every 8 hours", "as needed"],
    "instructions": ["food", "water", "milk", "meals"],
    "street_num": ["123", "456", "789", "1001"],
    "street_name": ["Main Street", "Oak Avenue", "Park Boulevard", "Elm Road"],
    "city": ["Springfield", "Riverside", "Greenville", "Portland"],
    "state": ["CA", "NY", "TX", "FL"],
    "zipcode": ["90210", "10001", "75001", "33101"],
    "phone": ["555-1234", "555-5678", "555-9012"],
    "extension": ["101", "205", "309"],
    "email_local": ["john.doe", "info", "support", "contact"],
    "email_domain": ["example", "company", "service"],
    "month": ["November", "December", "January", "February"],
    "day": ["17th", "25th", "1st", "15th"],
    "year": ["2025", "2026"],
    "time": ["9:30 AM", "2:15 PM", "6:45 PM"],
    "item": ["Coffee", "Sandwich", "Notebook", "Pen"],
    "qty": ["1", "2", "3"],
    "price": ["4.50", "8.99", "12.00"],
    "subtotal": ["4.50", "17.98", "36.00"],
    "name": ["John Doe", "Sarah Smith", "Michael Johnson", "Emily Davis", "David Wilson"],
    "color": ["blue", "red", "black", "white", "green"],
    "accessory": ["backpack", "hat", "glasses", "scarf"],
    "obstacle": ["obstacle", "barrier", "pole", "sign", "person"],
    "hazard": ["wet floor", "debris", "pothole", "broken glass"],
    "action_warn": ["stop", "step back", "turn around", "proceed slowly"],
}

def generate_response(category: str, template: str) -> str:
    """Generate a response from a template by filling in vocabulary."""
    response = template
    # Find all placeholders in the template
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    for placeholder in placeholders:
        if placeholder in VOCABULARY:
            value = random.choice(VOCABULARY[placeholder])
            response = response.replace(f'{{{placeholder}}}', value, 1)
    
    return response

def calculate_word_count(text: str) -> int:
    """Calculate word count of a text."""
    return len(text.split())

def generate_corpus(total_samples: int = 1000) -> List[Dict]:
    """Generate the complete corpus of responses."""
    
    # Define distribution
    distribution = {
        "scene_description": 200,
        "object_location": 200,
        "navigation": 200,
        "document_reading": 150,
        "person_recognition": 150,
        "warning": 100,
    }
    
    corpus = []
    response_id = 1
    
    for category, count in distribution.items():
        templates = RESPONSE_TEMPLATES[category]
        
        for _ in range(count):
            # Select random template
            template = random.choice(templates)
            
            # Generate response
            text = generate_response(category, template)
            
            # Calculate metrics
            word_count = calculate_word_count(text)
            
            # Determine length category
            if word_count <= 10:
                length_category = "short"
            elif word_count <= 25:
                length_category = "medium"
            elif word_count <= 50:
                length_category = "long"
            else:
                length_category = "very_long"
            
            # Create response entry
            response_entry = {
                "response_id": f"RESP_{response_id:04d}",
                "category": category,
                "text": text,
                "word_count": word_count,
                "length_category": length_category,
                "template": template
            }
            
            corpus.append(response_entry)
            response_id += 1
    
    # Shuffle corpus for diversity
    random.shuffle(corpus)
    
    return corpus

def save_corpus(corpus: List[Dict], output_dir: Path):
    """Save corpus to CSV and generate metadata."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    csv_path = output_dir / "response_corpus.csv"
    fieldnames = ["response_id", "category", "text", "word_count", "length_category", "template"]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corpus)
    
    print(f"✓ Saved corpus to {csv_path}")
    
    # Generate statistics
    stats = {
        "total_responses": len(corpus),
        "categories": {},
        "length_categories": {},
        "word_count_stats": {
            "min": min(r["word_count"] for r in corpus),
            "max": max(r["word_count"] for r in corpus),
            "mean": sum(r["word_count"] for r in corpus) / len(corpus),
        }
    }
    
    # Category distribution
    for category in set(r["category"] for r in corpus):
        stats["categories"][category] = sum(1 for r in corpus if r["category"] == category)
    
    # Length distribution
    for length_cat in ["short", "medium", "long", "very_long"]:
        stats["length_categories"][length_cat] = sum(1 for r in corpus if r["length_category"] == length_cat)
    
    # Save statistics
    stats_path = output_dir / "metadata" / "text_statistics.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved statistics to {stats_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CORPUS GENERATION SUMMARY")
    print("="*60)
    print(f"Total Responses: {stats['total_responses']}")
    print(f"\nCategory Distribution:")
    for cat, count in stats['categories'].items():
        print(f"  {cat:25} {count:4d} ({count/stats['total_responses']*100:.1f}%)")
    
    print(f"\nLength Distribution:")
    for length, count in stats['length_categories'].items():
        print(f"  {length:12} {count:4d} ({count/stats['total_responses']*100:.1f}%)")
    
    print(f"\nWord Count Statistics:")
    print(f"  Min:  {stats['word_count_stats']['min']:5.1f} words")
    print(f"  Max:  {stats['word_count_stats']['max']:5.1f} words")
    print(f"  Mean: {stats['word_count_stats']['mean']:5.1f} words")
    print("="*60 + "\n")

def main():
    """Main execution function."""
    print("Generating LUMENAA Response Corpus...")
    print("="*60)
    
    # Generate corpus
    corpus = generate_corpus(total_samples=1000)
    
    # Save corpus and metadata
    output_dir = Path(__file__).parent.parent / "data"
    save_corpus(corpus, output_dir)
    
    print("✓ Corpus generation complete!")
    print(f"✓ Output: {output_dir / 'response_corpus.csv'}")

if __name__ == "__main__":
    main()
