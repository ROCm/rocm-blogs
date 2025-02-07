GISCUS_PATH = "./scripts/html/giscus.html"

def giscus():
    with open(GISCUS_PATH, "r") as f:
        html = f.read()

    return html