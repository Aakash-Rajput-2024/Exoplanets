import subprocess
import json
import os
import sys

# Define the papers we want to query
papers = [
    {
        "arxiv_id": "1501.01332",
        "title": "Causal inference using invariant prediction: identification and confidence intervals",
        "author": "Peters et al.",
        "year": 2016
    },
    {
        "arxiv_id": "1907.02893",
        "title": "Invariant Risk Minimization",
        "author": "Arjovsky et al.",
        "year": 2019
    },
    {
        "arxiv_id": "2003.00688",
        "title": "Out-of-Distribution Generalization via Risk Extrapolation (REx)",
        "author": "Krueger et al.",
        "year": 2020
    },
    {
        "arxiv_id": "2006.06485",
        "title": "Deep Structural Causal Models for Tractable Counterfactual Inference",
        "author": "Pawlowski et al.",
        "year": 2020
    },
    {
        "arxiv_id": "2007.01434",
        "title": "In Search of Lost Domain Generalization",
        "author": "Gulrajani & Lopez-Paz",
        "year": 2020
    },
    {
        "arxiv_id": "2010.05761",
        "title": "The Risks of Invariant Risk Minimization",
        "author": "Rosenfeld et al.",
        "year": 2021
    },
    {
        "arxiv_id": "2102.11107",
        "title": "Towards Causal Representation Learning",
        "author": "Schölkopf et al.",
        "year": 2021
    },
    {
        "arxiv_id": "2406.13371",
        "title": "Identifiable Causal Representation Learning: Unsupervised, Multi-View, and Multi-Environment",
        "author": "von Kügelgen",
        "year": 2024
    },
    {
        "arxiv_id": "make-01-00019-v2",
        "title": "Causal Discovery with Attention-Based Convolutional Neural Networks",
        "author": "Nauta et al.",
        "year": 2019
    }
]

cli_path = "/Users/aakashrajput/.gemini/config/plugins/science/skills/literature_search_openalex/scripts/openalex_cli.py"

results = []

for paper in papers:
    print(f"Querying OpenAlex for: {paper['title']}...")
    try:
        # Search by exact title using OpenAlex CLI
        cmd = [
            "uv", "run", cli_path,
            "filter", "works",
            "--search", paper['title'],
            "--per-page", "3",
            "--select", "id,display_name,cited_by_count,publication_year,doi"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        
        # Try to find a good match in the results
        matched = False
        if "results" in data and len(data["results"]) > 0:
            for item in data["results"]:
                # Compare title similarity or year
                title_lower = item.get("display_name", "").lower()
                target_lower = paper["title"].lower()
                # Simple exact match or subset match
                if target_lower in title_lower or title_lower in target_lower or item.get("publication_year") == paper["year"]:
                    results.append({
                        "arxiv_id": paper["arxiv_id"],
                        "title": item.get("display_name"),
                        "author": paper["author"],
                        "year": item.get("publication_year"),
                        "citations": item.get("cited_by_count", 0),
                        "doi": item.get("doi"),
                        "openalex_id": item.get("id")
                    })
                    matched = True
                    print(f"  Matched: '{item.get('display_name')}' with {item.get('cited_by_count')} citations")
                    break
            
            if not matched:
                # If no exact match but we have results, take the first one
                first_item = data["results"][0]
                results.append({
                    "arxiv_id": paper["arxiv_id"],
                    "title": first_item.get("display_name"),
                    "author": paper["author"],
                    "year": first_item.get("publication_year"),
                    "citations": first_item.get("cited_by_count", 0),
                    "doi": first_item.get("doi"),
                    "openalex_id": first_item.get("id")
                })
                print(f"  Fallback matched: '{first_item.get('display_name')}' with {first_item.get('cited_by_count')} citations")
        else:
            print(f"  No results found for {paper['title']}")
            results.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "author": paper["author"],
                "year": paper["year"],
                "citations": 0,
                "doi": None,
                "openalex_id": None
            })
    except Exception as e:
        print(f"  Error querying {paper['title']}: {e}")
        results.append({
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "author": paper["author"],
            "year": paper["year"],
            "citations": 0,
            "doi": None,
            "openalex_id": None
        })

# Sort by citations descending
sorted_results = sorted(results, key=lambda x: x["citations"], reverse=True)

print("\n=== SORTED RESULTS ===")
for r in sorted_results:
    print(f"{r['citations']} citations | {r['title']} ({r['author']}, {r['year']}) | arXiv: {r['arxiv_id']}")

# Write to a JSON file
with open("/Users/aakashrajput/MachineLearning/Exoplanets/scratch/citations_sorted.json", "w") as f:
    json.dump(sorted_results, f, indent=2)
