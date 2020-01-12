# -*- encoding: utf-8 -*-

#Programa  : desplura.py
#Finalidade: Este programa recebe uma palavra em portugues, verifica se ela é  um substantivo ou um adjetivo ou um advérbio.
#            Caso seja uma destas tres classes gramáticais o programa verifica se a palavra está no plural e se estiver 
#            retorna o singular da mesma.
#Data      : 2020.01;12
#Autor     : OmaR

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='portuguese')
portSte = PorterStemmer()


import pattern
from pattern.en import singularize
import spacy
nlp = spacy.load('pt')



def Singular(palavra):
    if palavra.find('-') > -1:
        return (False, '')
        
    texto = nlp(palavra)
    arTexto = [(i.pos_, i.dep_) for i in texto if i.pos_ in ('NOUN', 'ADV', 'ADJ')]
    if len(arTexto) == 0:
        #print([(i.pos_, i.dep_) for i in texto])
        #print('Esta palavra não é substantivo ou adjetivo ou adverbio')
        return (False,'')
    
    if palavra[-3:] == 'ões' or palavra[-3:] == 'ãos' or palavra[-3:] == 'ães':
        troca = 'ão'      
        palavra = palavra[:-3] + 'ão'
        
    elif palavra[-3:] == 'eis':
        palavra = palavra[:-3] + 'el'
        
    elif palavra[-2:] == 'es':
        palavra = palavra[:-2]
        
    elif palavra[-2:] == 'ns':
        palavra = palavra[:-2] + 'm'
        
    elif palavra[-1:] == 's':
        palavra = palavra[:-1]
    #print('{} Singular = {}'.format(arTexto, palavra))
    return (True, palavra)
    
    
    
fArq = open('dicionario.txt', 'r', encoding='utf-8').readlines()


fRes = []
for it in fArq:
    
    if it[-2:-1].lower() == 's':
        res = Singular(it[:-1])
        
        if stemmer.stem(it[:-1])[:-1] ==  stemmer.stem(res[1])[:-1]:
            fRes.append(res[1])
            print('Plural;{};singular;{}'.format(it[:-1], res[1]))
            
            
