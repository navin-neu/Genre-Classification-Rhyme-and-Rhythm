# NAVIN KUMAR, SUNG JUN LEE, NAMDAR KABOLINEJAD 2021

# LyricGenius scraper and cleaner as described in report.
# Running this will populate the CSV Lyrics folder with a fully cleaned dataset in CSV format.
# We omit the lyrics themselves from this repo to avoid any potential copyright infringement.

# Scraper functions written by Navin Kumar
# Data cleaning functions written by Namdar Kabolinejad
# CSV conversion functions written by Sung Jun Lee

### NOTE ###
# This script requires the installation of pip packages lyricsgenius and langdetect.
# It is also necessary to integrate a fix for lyricsgenius described in the report.

import lyricsgenius
import re
import pandas as pd
from nltk import sent_tokenize
from langdetect import detect


token = "" #Paste API token here
genius = lyricsgenius.Genius(token)
genres = ['rap', 'pop', 'r-b', 'rock', 'country']

def get_lyrics_from_genius(num_pages, genre):
    page = 1
    lyrics = []
    while page:
        while True:
            try: #for soome reason the genius api likes to timeout, so we just brute force this...
                res = genius.tag(genre, page=page)
                for hit in res['hits']:
                    song_lyrics = genius.lyrics(song_url=hit['url'])
                    lyrics.append(song_lyrics)
                break
            except:
                pass    
        page = res['next_page']
        if page == num_pages: break

    with open(genre + '.txt', 'w') as f:
        for i in lyrics:
            f.write(i)
            f.write('\n')


def read_in_file(file_name, data_path = "."):
    path = data_path + "/" + file_name + ".txt"    
    try:
        with open (path, "r") as file:
            data = file.read().splitlines()
            return data
    except:
        print("[Error] can't load file, singer doesn't exist")


def write_data(file_name, data_array):
    with open(file_name+".txt", "w") as txt_file:
        for line in data_array:
            txt_file.write(line+"\n") 


def clean_data(data_array, punc_list='[^\w\s]', in_parenth=True, in_brackets=True, in_curly=True,
               header=True, empty_line=True, punctuation=True):
    rtn_array = []
    remove = []
    
    for i, line in enumerate(data_array):
        new_line = line
        
        if(in_parenth):
            new_line = re.sub(r"\([^()]*\)", "", new_line)
            
        if(in_brackets):
            new_line = re.sub(r"\[[^()]*\]", "", new_line)
            
        if(in_curly):
            new_line = re.sub(r"\{[^()]*\}", "", new_line)
        
        if(punctuation):
            new_line = re.sub(r'[^\'\w\s]', '', new_line)
            
        rtn_array.append(new_line)
    
    for i, line in enumerate(rtn_array):
        if(empty_line):
            if (len(line) < 2):
                remove.append(i)
                
        if(header):
            if (len(line) > 1) and ((line[0] == '\t') or (line[0] == ' ')):
                remove.append(i)

    for index in sorted(list(set(remove)), reverse=True):
        del rtn_array[index]
    
    return(rtn_array)


def write_to_csv(label, genrename):

    def filter_conditions(lyric):

        nlines = len(sent_tokenize(lyric))
        if nlines < 4 or nlines > 20: return False
        
        lang = detect(lyric)
        if lang != "en": return False

        return True

    with open(genrename + ".txt", "r") as f:
        f.seek(0)
        paragraphs = []
        paragraph = ""
    
        for line in f:
            if line == "\n" and paragraph != "":
                paragraphs.append(paragraph)
                paragraph = ""
            elif line != "\n":
                paragraph += line.lower().replace('\n', "").strip()+". "

        filtered_paragraphs = list(filter(filter_conditions, paragraphs))

        genre_id = [label for i in range(len(filtered_paragraphs))]
        genre = [genrename for i in range(len(filtered_paragraphs))]
        df = pd.DataFrame({'lyrics': filtered_paragraphs, 'genre_id': genre_id, "genre": genre}) 
        df.to_csv(genrename+".csv")
        


#Run the scraper for each genre
label = 1
for genre in genres:
    get_lyrics_from_genius(25, genre)
    genre_to_clean = read_in_file(genre)
    genre_cleaned = clean_data(genre_to_clean, empty_line=False, in_parenth=False)
    write_data(genre, genre_cleaned)
    write_to_csv(label, genre)
    label += 1






