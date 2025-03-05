import requests
import pandas as pd
from fake_useragent import UserAgent


def get_refs_and_cites(doi, cr):
    cites = opencitations(doi)
    refs = crossref_references(doi, cr)
    return refs, cites

def deduplicate_edges(edges):
    return list(set(edges))

def recursive_graph(base_doi: str, edges: list, doi: str = "", depth=3):
    """
    Build the citation graph, starting from doi, with depth.
    Need to deduplicate edges after.
    Hoping that fixing the depth to 5 would yield a single connected component for multiple starting dois
    """
    if depth == 0:
        return
    # time.sleep(0.5)
    print(depth)
    d = doi if len(doi) else base_doi
    refs, cites = get_refs_and_cites(d)
    refs_edges = zip([d] * len(refs), refs)
    cites_edges = zip(cites, [d] * len(cites))
    edges.extend(refs_edges)
    edges.extend(cites_edges)
    print(f"Number of edges: {len(edges)}")
    for ref in refs:
        recursive_graph(base_doi, edges, ref, depth-1)
    if depth < 3:
        for cite in cites:
            if cite != base_doi:
                recursive_graph(base_doi, edges, cite, depth-1)


def crossref_references(doi, cr):
    try:
        work = cr.works(ids=doi)
    except Exception as e:
        print(f"In crossref got an exception {e}")
        return []
    references = work['message'].get('reference', [])
    reference_dois = []
    for ref in references:
        if 'DOI' in ref:
            reference_dois.append(ref['DOI'])
    return reference_dois

def opencitations(doi):
    ua = UserAgent()
    open_citations_token = "a3409f07-fd49-4230-b1fb-5114ff0a5c52"

    base_url = "https://opencitations.net/index/coci/api/v1/citations/"

    headers = {"Accept": "application/json",
               "Authorization": f"{open_citations_token}",
               "User-Agent": str(ua.random)}
    try:
        response = requests.get(f"{base_url}{doi}", headers=headers, timeout=4)
        response.raise_for_status()
        citations = response.json()
        return [entry['citing'] for entry in citations] if citations else []
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(e)
    return []

# def get_article_info(doi):
#     url = f"https://api.crossref.org/works/{doi}.xml"
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         # Parse the XML to extract the abstract
#         import xml.etree.ElementTree as ET
#         root = ET.fromstring(response.content)
        
#         # Find the abstract element
#         abstract_element = root.find(".//{http://www.crossref.org/schema/4.3.0}abstract")
        
#         if abstract_element is not None:
#             return abstract_element.text
#         else:
#             return "Abstract not found."
#     else:
#         return "Failed to retrieve abstract."



# if __name__ == "__main__":
#     doi = "10.1007/JHEP07(2017)107"
#     res = get_article_info(doi)
#     print(res)