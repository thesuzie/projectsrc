import pandas as pd
import re

txt = pd.read_csv('women20.csv', header=0, sep=',')
# names=['IdPost','Author','Thread','Timestamp','Content','AuthorNumPosts','AuthorReputation','LastParse','parsed','Site','CitedPost','AuthorName','Likes'])

replacements = ['IMG', 'QUOTE', 'CODE', 'CITING', 'LINK', 'IFRAME']


def remove_replacements(post_content):
    content = post_content
    pattern = '\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*.*\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*'
    cleaned = re.sub(pattern,' ', content)
    return cleaned


contentDF = txt.drop(['Author','Thread','Timestamp','AuthorNumPosts','AuthorReputation','LastParse','parsed','Site','CitedPost','AuthorName','Likes'], axis=1)

removedlinks = [remove_replacements(c) for c in contentDF['Content']]
contentDF['Content Cleaner'] = removedlinks

print(contentDF.head())

contentDF.to_csv('women20_cleaned.csv')
