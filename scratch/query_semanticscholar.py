import urllib.request
import json
import time

papers = [
    {"arxiv_id": "1501.01332", "name": "invariant prediction"},
    {"arxiv_id": "1907.02893", "name": "IRM"},
    {"arxiv_id": "2003.00688", "name": "REx"},
    {"arxiv_id": "2006.06485", "name": "DeepSCM"},
    {"arxiv_id": "2007.01434", "name": "DomainBed"},
    {"arxiv_id": "2010.05761", "name": "Risks of IRM"},
    {"arxiv_id": "2102.11107", "name": "Towards CRL"},
    {"arxiv_id": "2406.13371", "name": "Identifiable CRL"},
    {"doi": "10.3390/make1010019", "name": "Causal Discovery Attention CNN"}
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

results = []

for paper in papers:
    if "arxiv_id" in paper:
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper['arxiv_id']}?fields=title,citationCount,year,authors,externalIds,doi"
    else:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{paper['doi']}?fields=title,citationCount,year,authors,externalIds,doi"
        
    print(f"Querying Semantic Scholar for {paper['name']} ({url})...")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            results.append({
                "source_id": paper.get("arxiv_id") or paper.get("doi"),
                "title": data.get("title"),
                "year": data.get("year"),
                "citations": data.get("citationCount", 0),
                "doi": data.get("doi"),
                "authors": [a["name"] for a in data.get("authors", [])]
            })
            print(f"  Success: {data.get('title')[:50]}... | {data.get('citationCount')} citations")
    except Exception as e:
        print(f"  Error: {e}")
        # Fallback to searching by name
        results.append({
            "source_id": paper.get("arxiv_id") or paper.get("doi"),
            "title": paper["name"],
            "year": None,
            "citations": -1,
            "doi": None,
            "authors": []
        })
    time.sleep(1) # Be polite

# Print and save
sorted_results = sorted(results, key=lambda x: x["citations"], reverse=True)
print("\n=== SEMANTIC SCHOLAR RESULTS ===")
for r in sorted_results:
    print(f"{r['citations']} citations | {r['title']} ({', '.join(r['authors'][:2])} et al., {r['year']}) | {r['source_id']}")

with open("/Users/aakashrajput/MachineLearning/Exoplanets/scratch/citations_semanticscholar.json", "w") as f:
    json.dump(sorted_results, f, indent=2)
