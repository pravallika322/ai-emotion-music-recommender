import webbrowser
import urllib.parse

def play_on_youtube(keyword):
    """
    Opens YouTube search results for the given keyword
    """
    query = urllib.parse.quote(keyword)
    url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(url)
