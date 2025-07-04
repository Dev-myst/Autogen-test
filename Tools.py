import arxiv
def arxiv_search(query: str, max_results: int = 3) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[dict] = []

    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "authors": [x.name for x in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
            }
        )
    return papers
