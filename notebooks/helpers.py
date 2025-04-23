import requests
import pandas as pd
from fake_useragent import UserAgent
from dotenv import load_dotenv
import os
from os.path import join, dirname
from habanero.crossref import Crossref
from scholarly import scholarly

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

SCOPUS_KEY = os.environ.get("SCOPUS_API")

cr = Crossref()

def handle_empty(field, placeholder="unknown"):
    return field if field else placeholder

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

    base_url = "https://opencitations.net/index/meta/api/v1/citations/"

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


def keywords(doi):
    url = f"https://api.elsevier.com/content/search/scopus?query=DOI({doi})"
    headers = {"X-ELS-APIKey": SCOPUS_KEY, "Accept": "application/json"}
    response = requests.get(url, headers=headers)

    metadata = response.json()
    keywords = metadata["search-results"]["entry"][0].get("authkeywords", [])
    return keywords if keywords else None

def keywords_scholar(doi):

    search_query = scholarly.search_pubs(doi)
    paper = next(search_query)

    keywords = paper.get("bib", {}).get("keywords", None)
    return keywords


def metadata(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    metadata = response.json()["message"]

    # Extracting details
    title = metadata.get("title", [""])[0]
    authors = [f"{author['given']} {author['family']}" for author in metadata.get("author", [])]
    journal = metadata.get("container-title", [""])[0]
    published_date = metadata.get("published-print", {}).get("date-parts", [[""]])[0]
    abstract = metadata.get("abstract", "")
    return title, authors, journal, published_date, abstract

def metadata_scholar(doi):
    search_query = scholarly.search_pubs(doi)
    paper = next(search_query)  # Get the first result
    title = paper.get('bib', {}).get("title", None)
    authors = paper.get('bib', {}).get("author", None)
    journal = paper.get('bib', {}).get('journal', None)
    year = paper.get('bib', {}).get("pub_year", None)
    abstract = paper.get('bib', {}).get("abstract", None)
    return title, authors, journal, year, abstract


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

def get_all_metadata(doi):
    meta = None
    keys = None
    try:
        meta = metadata(doi)
    except Exception as _:
        pass
    # if meta is None:
    #     try:
    #         meta = metadata_scholar(doi)
    #     except Exception as _:
    #         pass
    try:
        keys = keywords(doi)
    except Exception as _:
        pass
    # if keys is None:
    #     try:
    #         keys = keywords_scholar(doi)
    #     except Exception as _:
    #         pass
    return {'title': meta[0] if meta else None, "authors": meta[1] if meta else None,
             "venue": meta[2] if meta else None, "year": meta[3] if meta else None, "abstract": meta[4] if meta else None, "keywords": keys}


if __name__ == "__main__":
    doi = "10.1143/JJAP.27.L209"
    print(get_all_metadata(doi))
