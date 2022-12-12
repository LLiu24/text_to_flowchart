import altair as alt
import pandas as pd


def viz(csvfilename):
    data = pd.read_csv(csvfilename)
    bars = alt.Chart(data).mark_bar().encode(
        x = 'count(entity_lemma_label):Q',
        y = 'entity_lemma_label:O',
        color = alt.Color('entity_lemma_label', scale=alt.Scale(scheme = 'viridis'))).interactive()

    data_gb = data.groupby('entity_lemma_label').count()
    data_gb.reset_index(inplace=True)

    pies = alt.Chart(data_gb).mark_arc().encode(
        theta=alt.Theta(field='entity_lemma_text', type='quantitative'),
        color=alt.Color(field='entity_lemma_label', type='nominal', scale=alt.Scale(scheme = 'viridis')))
    
    return bars.display(),pies.display()




