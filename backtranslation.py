#################################################################################
# IMP NOTE: Export the json file linked with your Google account on Google Cloud 
# otherwise the service won't work since it is a paid service
# Refer: https://cloud.google.com/translate/docs/basic/setup-basic


from google.cloud import translate_v2 as translate
import pandas as pd

translate_client = translate.Client()

df = pd.read_csv("data/train_baseline.csv")

def fun(inp):
    global src
    global target
    result = translate_client.translate(inp, target_language=target, source_language=src)
    return result['translatedText']

src = 'en'
target = 'es'
df['augmented_text'] = df['comment_text'].apply(fun)

# print(df.head(3))

src = 'es'
target = 'en'
df['comment_text'] = df['augmented_text'].apply(fun)

# print(df.head(3))

df.to_csv("data/train_baseline_"+src+'.csv', index=False)

print('done')