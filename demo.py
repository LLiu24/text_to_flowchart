#!/usr/bin/env python
# coding: utf-8

# # Translate text into flowchart

# Q: distinguish decision & non-decision related task?


# ## import data
import os
import spacy
from spacy import displacy
from spacy.pipeline import EntityRecognizer
import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from first_n_words import first_n_words
sns.set()



# ## Identify Named Entity

# ### Identify verb especially decision related verb


def text_to_flowchart(example):
    text_title = list(example.keys())[0]
    text = example[text_title]
    text = re.sub(r'\n|\r',' ',text)
    text = re.sub(r'\s{2}',' ',text)
    text = re.sub(r'\d','',text)
    # Build upon the spaCy small model
    nlp = spacy.load('en_core_web_sm')

    # Create the Ruler and Add it
    ruler = nlp.add_pipe('entity_ruler',before='ner')
    
    # Define decision words. Currently hard-coded based on web search. Can be improved by ML
    
    decision_words=['specify','decide','agree','determine',\
                    'need','conclude','settle','resolve','commit','adjudicate','verify','validate','ensure']


    # List of Entities and Patterns (source: https://spacy.io/usage/rule-based-matching)
    pattern_decision = [
                    {'label':'DECISION',
                     'pattern':[
                                 {'POS':{'IN':['ADJ','NOUN','PART','PRON','SCONJ','ADV']},'OP':'{0,}'},
                                 {'LEMMA':{'IN':decision_words}},
                                 {'ORTH':',','OP':'{0,1}'},
                                 {'POS':{'IN':['ADJ','NOUN','PART','DET','ADP',\
                                               'VERB','PRON','CCONJ']},'OP':'{0,}'},
                                 {'ORTH':',','OP':'{0,1}'},
                                 {'POS':{'IN':['ADJ','NOUN','PART','DET','ADP','PRON','CCONJ']},'OP':'{0,}'}
                               ]}
                   ]


    pattern_root = [
                    {'label':'ACTION',
                     'pattern':[
                                 {'POS':{'IN':['ADJ','NOUN','PART','PRON','SCONJ','ADV']},'OP':'{0,}'},
                                 {'DEP':'ROOT','POS':'VERB','LEMMA':{'NOT_IN':decision_words}},
                                 {'ORTH':',','OP':'{0,1}'},
                                 {'POS':{'IN':['ADJ','NOUN','PART','DET','ADP',\
                                               'VERB','PRON','CCONJ']},'OP':'{0,}'},
                                 {'ORTH':',','OP':'{0,1}'},
                                 {'POS':{'IN':['ADJ','NOUN','PART','DET','ADP','PRON','CCONJ']},'OP':'{0,}'}
                               ]}
                   ]



    #Add patterns to ruler
    ruler.add_patterns(pattern_decision)

    ruler.add_patterns(pattern_root)

    #Create the doc
    doc = nlp(text)

    # ### Create a DataFrame to store named entity label and text

    extraction_ls = []

    for ent in doc.ents:
        if ent.label_ =='DECISION' or ent.label_ =='ACTION':
            entity_lemma_label = ent.label_
            entity_lemma_text = ent.text.lower()
            extraction_ls.append({'entity_lemma_label':entity_lemma_label,'entity_lemma_text':entity_lemma_text})
        else:
            pass


    extraction_df = pd.DataFrame(extraction_ls)
    
    # save extraction_df to csv
    extraction_df.to_csv(text_title+'.csv',index = False)
    
        # ### Create relationship for actions

    source = [d for d in extraction_df['entity_lemma_text']]
    target = [i for i in source[1:]]
    edge = [i for i in extraction_df['entity_lemma_label'][:-1]]

    zipped = zip(source,target,edge)

    relationship_df = pd.DataFrame(zipped,columns=['source','target','edge'])

    relationship_df['edge_value'] = 1


        # ### Graph analysis and visualization

    G = nx.Graph()

    G = nx.from_pandas_edgelist(relationship_df,
                            source='source',
                            target='target',
                            edge_attr='edge_value',
                            edge_key='edge',
                            create_using=nx.Graph())
    
    #Extract first 4 words from each node for flowchar
    n = 4
    start_end_nodes = [first_n_words(list(G.nodes)[0],n),
                   first_n_words(list(G.nodes)[-1],n)]

    decision_nodes = [first_n_words(item,n) for i,item in enumerate(list(G.nodes)) for decision_word in \
                      decision_words if decision_word in item and i not in [0,-1]]


    other_nodes = [first_n_words(i,n) for i in list(G.nodes) for decision_word in decision_words \
                   if decision_word not in i and first_n_words(i,n) not in start_end_nodes and first_n_words(i,n) not in decision_nodes]

    node_color_list = []
    node_shape_list = []
    for i in list(G.nodes):
        if i in start_end_nodes:
            node_color_list.append('#323050')
            node_shape_list.append('o')
        elif i in decision_nodes:
            node_color_list.append('#2A5159')
            node_shape_list.append('D')
        else:
            node_color_list.append('#323050')
            node_shape_list.append('s')

#     options = {
#         'node_color': node_color_list,
#         'node_size':4000,
#         'node_shape':'s',
#         'alpha':0.8,
#         'width':0.4,
#         'edge_cmap':plt.cm.Blues,
#         'font_weight':'bold',
#         'font_size':16,
#         'font_color':'white',
#         'with_labels':True
#     }


    plt.figure(figsize=(20,20))
    pos = dict((first_n_words(ele,n),(0.5,-i)) for i,ele in enumerate(list(G.nodes)))
    node_color_start_end = '#323050'
    node_color_decision = '#05A0FA'
    node_color_other = '#323050'
    nodesize = 20000
    shape_legend = {'o':'start or end action','D':'decision','s':'in-between action'}
    # pos = nx.spring_layout(G)
    # nx.draw(G,pos,**options)
    nx.draw_networkx_nodes(G,pos,nodelist=start_end_nodes,node_color=node_color_start_end,\
                           node_size=nodesize,node_shape='o')
    nx.draw_networkx_nodes(G,pos,nodelist=decision_nodes,node_color=node_color_decision,\
                           node_size=nodesize*0.7,node_shape='D')
    nx.draw_networkx_nodes(G,pos,nodelist=other_nodes,node_color=node_color_other,\
                           node_size=nodesize,node_shape='s')
    nx.draw_networkx_labels(
        G,
        pos,
        labels={first_n_words(node,n):first_n_words(node,n) for node in list(G.nodes)},
        font_size=14,
        font_color='white',
        font_family='sans-serif',
        font_weight='normal',
        alpha=None,
        bbox=None,
        horizontalalignment='center',
        verticalalignment='top',
        ax=None,
        clip_on=True,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(first_n_words(a,n),first_n_words(b,n)) for a,b in G.edges()],
        width=1.0,
        edge_color='k',#[node_color_start_end]*2+[node_color_decision]+[node_color_other]*4,
        style='solid',
        alpha=None,
        arrowstyle='fancy',
        arrowsize=10,
        edge_cmap=None,
        edge_vmin=None,
        edge_vmax=None,
        ax=None,
        arrows=True,
        label=True, #None or True,
        node_size=300,
        nodelist=None,
        node_shape='o',
        connectionstyle='arc3',
        min_source_margin=0,
        min_target_margin=0,
    )

    legend_elements = [Line2D([0],[0],color = node_color_start_end,\
                    marker='o',markersize=30,label=shape_legend['o'],markevery=40),\
                       Line2D([0],[0],color = '#FFFFFF',alpha=0),
                       Line2D([0],[0],color = '#FFFFFF',alpha=0),
                       Line2D([0],[0],color = node_color_decision, \
                    marker='D',markersize=30,label=shape_legend['D'],markevery=40),\
                       Line2D([0],[0],color = '#FFFFFF',alpha=0),
                       Line2D([0],[0],color = '#FFFFFF',alpha=0),
                       Line2D([0],[0],color = node_color_other,\
                    marker='s',markersize=30,label=shape_legend['s'],markevery=40)]
    plt.legend(handles=legend_elements,loc='upper right',fontsize=20)
    plt.title(text_title,
              fontdict={'fontsize': 32,
                        'fontweight': 10,
                        'color': 'k',
                        'verticalalignment': 'baseline',
                        'horizontalalignment': 'center'})
    
    displacy_colors = {'ACTION': 'linear-gradient(90deg, #B2BEBF, #889C9B)','DECISION': 'linear-gradient(90deg, #73c5f5, #05A0FA)'}
    displacy_options = {'ents': ['ACTION','DECISION'], 'colors': displacy_colors}
    
    print(displacy.render(doc,style='ent',options=displacy_options))
    return plt.show()


