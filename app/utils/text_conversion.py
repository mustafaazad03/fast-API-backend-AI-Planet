import re

def convert_to_html(text):
    # Convert headers
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\n', r'<br>', text)
    text = re.sub(r'\* (.*?)<br>', r'<li>\1</li>', text)
    
    # Convert newlines and list items
    text = re.sub(r'\n', r'<br>', text)
    text = re.sub(r'\* (.*?)<br>', r'<li>\1</li>', text)

    # Handle nested lists (strong indicators for nested lists)
    text = re.sub(r'\*\* (\w)\. (.*?)<br>', r'<strong>\1.</strong> \2<br>', text)
    text = re.sub(r'\*\* (\d)\. (.*?)<br>', r'<strong>\1.</strong> \2<br>', text)
    text = re.sub(r'\*\* (\w\.) (.*?)<br>', r'<strong>\1</strong> \2<br>', text)

    # Handle <ul> and <ol> tags for list items
    text = re.sub(r'(<br>)(<li>)', r'\1<ul>\2', text)
    text = re.sub(r'(<li>.*?</li>)(<br>)(?=<li>)', r'\1\2', text)
    text = re.sub(r'(<li>.*?</li>)(<br>)(?!<li>)', r'\1</ul>\2', text)

    # Add <ol> for ordered lists
    text = re.sub(r'<strong>(\d\.)</strong> (.*?)<br>', r'<ol><li>\2</li></ol>', text)

    # Convert tables
    text = re.sub(r'\n\| (.*?) \| (.*?) \|\n', r'<tr><th>\1</th><th>\2</th></tr>', text)
    text = re.sub(r'\| (.*?) \| (.*?) \|\n', r'<tr><td>\1</td><td>\2</td></tr>', text)

    # Add <table> tags
    text = re.sub(r'<tr><th>', r'<table><thead><tr><th>', text)
    text = re.sub(r'</th><th>', r'</th><th>', text)
    text = re.sub(r'</th></tr><tr><td>', r'</th></tr></thead><tbody><tr><td>', text)
    text = re.sub(r'</td><td>', r'</td><td>', text)
    text = re.sub(r'</td></tr>', r'</td></tr>', text)
    text = re.sub(r'</tr>', r'</tr></tbody></table>', text)

    # Add <br> tags after colons, but handle repeated colons appropriately
    text = re.sub(r'(?<!:):(?!:)', r':<br>', text)

    # Add spaces around tags to avoid overlapping of tags
    tags = ['br', 'strong', 'li', 'ul', 'ol', 'tr', 'th', 'td', 'table', 'thead', 'tbody']
    for tag in tags:
        text = re.sub(rf'<{tag}>', rf' <{tag}> ', text)
        text = re.sub(rf'</{tag}>', rf' </{tag}> ', text)

    return text