import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    print(sys.argv)
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    finalDistribution = {}
    linkedPages = corpus[page]
    for item in corpus:
        currentProb = (1-damping_factor)/len(corpus)
        if len(linkedPages) > 0:
            if item in linkedPages:
                currentProb += (damping_factor/len(linkedPages))
        else:
            currentProb += (damping_factor/len(corpus))
        
        finalDistribution[item] = currentProb
        
    return finalDistribution
        
        
        


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sampledDict = {}
    for i in corpus: 
        sampledDict[i] = 0
    
    randomPage = random.choice(list(corpus.keys()))
    
    for count in range(n):
        sampledDict[randomPage] += (1/n)
        transMod = transition_model(corpus, randomPage, damping_factor)
        randomPage = random.choices(list(transMod.keys()), list(transMod.values()))[0]
        
    return sampledDict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    iteratedDict = {}
    for item in corpus:
        iteratedDict[item] = 1/len(corpus)
    
    while True:
        continuing = False
        temporaryDict = {}
        for item in corpus:
            prob = (1-damping_factor)/len(corpus)
            for sItem in corpus:
                if item != sItem:
                    if len(corpus[sItem]) > 0:
                        if(item in corpus[sItem]):
                            prob += damping_factor * (iteratedDict[sItem]/len(corpus[sItem]))
                    
                    else:
                        prob += damping_factor * (iteratedDict[sItem]/len(corpus))
            temporaryDict[item] = prob
            if abs(iteratedDict[item] - temporaryDict[item]) > 0.001:
                continuing = True
        
        if not continuing:
            return temporaryDict
        
        else:
            iteratedDict = temporaryDict
            
            
        
                


if __name__ == "__main__":
    main()
