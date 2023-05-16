# -*- coding: utf-8 -*-
"""
@author: Sangani
"""

#--------------------------------------------------------------------
# Import Library
# Connect Entrez to Email and API
#--------------------------------------------------------------------
from Bio import Entrez
import pandas as pd

Entrez.email = 'nsangani@iu.edu'
Entrez.api_key = '4ffdd38541ab011f0b7f7f8ee3b7fe123f09'

#--------------------------------------------------------------------
# Question 1.a : How many databases are included in the Entrez network? 
#--------------------------------------------------------------------
handle = Entrez.einfo()
record = Entrez.read(handle)
print(len(record['DbList']))

#--------------------------------------------------------------------
# Question 1.b : What are the names of those databases? 
#--------------------------------------------------------------------
for i in record['DbList']:
    print(i)

#--------------------------------------------------------------------
# Question 1.c : How many hits does it return if you query Pubmed for “CRISPR/Cas9”?
#--------------------------------------------------------------------

# handle = Entrez.einfo(db='pubmed')
# record = Entrez.read(handle)
handle = Entrez.esearch(db='pubmed', term='CRISPR/Cas9')
record = Entrez.read(handle)
print(record.keys())
print(record['Count']) #9093 
List = record['IdList'] #20 total
print(List)
#--------------------------------------------------------------------
# Question 1.d : Select three articles from the queried results. 
#Write a file that includes the article names, authors, ids, publish dates, and summaries. 
#--------------------------------------------------------------------

Ids = List[:3]
print(Ids)
with open ('CRISPR.txt', 'w', encoding="utf-8") as text:
    for ID in Ids: 
        
        handle = Entrez.esummary(db='pubmed', id = ID, retmode = 'xml')
        records = Entrez.parse(handle)
        for record in records:
            Title =  record['Title'] 
            Authors =  str(record['AuthorList']).strip('[]') 
            Id =  record['Id'] 
            PubDate =  record['PubDate'] 
        handle = Entrez.efetch(db = 'pubmed', id = ID, rettype = 'gb', retmode = 'text')
        summary = handle.read()
        handle.close()
        Next = '===================== Next Article ========================'
        
        text.write('Title: ' + Title + '\n')
        text.write('Authors: ' + Authors + '\n')
        text.write('ID: ' + Id + '\n')
        text.write('PubDate: ' + PubDate + '\n')
        text.write('Summary:' + '\n' + summary)
        text.write('\n' + Next + '\n')
    text.close()
        

#--------------------------------------------------------------------
# Question 2.a : Download the Strang Sequence file from Canvas
#--------------------------------------------------------------------

from Bio import SeqIO
from Bio.Blast import NCBIWWW

#--------------------------------------------------------------------
# Question 2.b : Perform a BLAST search using the sequence.
#--------------------------------------------------------------------

# Seq_Record = SeqIO.parse('Sequence.fasta', 'fasta')

Sequence ='cgatgattgttctctactaatcacaaggatattggtactttatacttactatttggggcatgagccggcatagta'
result_handle = NCBIWWW.qblast('blastn', 'nt', Sequence)

with open ('Blast_Results.txt', 'w') as file:
    file.write(result_handle.read())
    file.close()

#--------------------------------------------------------------------
# Question 2.c : Based upon your results, what is the likely source of the strange sequence? 
#--------------------------------------------------------------------

# Canis lupus laniger aka Tibetan wolf
# Canis lupus chanco  aka Himalayan/Mongolian wolf
# Canis aureus isolate CAU_Kenya_2 aka Golden Jackel
# & Others....

#  Closely related to Gray wolf (Canis lupus) in general. Too hard to read from txt file tbh

#--------------------------------------------------------------------
# Question 2.d : How did this occur? 
#--------------------------------------------------------------------

# Complete 100% alignment with that specific genome with no gaps.
# Not exactly sure what you mean by how it occurs...do you mean different species with similar sequence trend?
# There must be some sort of epigenetics effect that might have altered certain bp resulting speciation. 

#--------------------------------------------------------------------
# Question 2.d : Translate the sequence, and then perform a protein blast on that sequence. 
#--------------------------------------------------------------------

from Bio.Seq import Seq 

dna = Seq(Sequence)
Rna = dna.translate()
print(Rna)

blast_res = NCBIWWW.qblast('blastp', 'refseq_protein', Rna)

with open ('Blast_protein.txt', 'w') as file:
    file.write(blast_res.read())
    file.close()

#--------------------------------------------------------------------
# Question 2.d : Explain your result. 
#--------------------------------------------------------------------

# Protein cytochrome c oxidase subunit I in Pleurocorallium inutile species 100% overlap with 72% query cover with least background noise (E value)
# Protein cytochrome oxidase subunit 1 in Chondrodactylus angulifer species 91.30% overlap with 92% query cover. 

# This sequence encodes protein that is also seen in coral and reptile. 
# Other closely related proteins were from more reptiles and mammals. This clearly shows the evolutionary change and speciation!
# It is amazing to see how complex species like gray fox is yet another sub sub branch of coral's evolutionary tree.
# Could also be simply a mutation?