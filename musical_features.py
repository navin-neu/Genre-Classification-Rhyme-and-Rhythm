# NAVIN KUMAR 2021
# This script generates rhythmic and rhyming based musical features
# These are returned as a vector which can be appended onto a normal
# BoW feature vector.

###NOTE:####
#Before running this script, run:
#pip3 install symspellpy
#pip3 install pronouncing
#pip3 install jaro-winkler

from nltk.tokenize import sent_tokenize, word_tokenize
import pronouncing
from symspellpy import SymSpell, Verbosity
import pkg_resources
import itertools
import jaro

#To be run whenever script is imported:
sym_spell = SymSpell()
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, 0, 1)

#see https://github.com/hyperreality/Poetry-Tools/blob/master/poetrytools/poetics.py
STANDARD_METERS = {
    'iambic trimeter': '010101',
    'iambic tetrameter': '01010101',
    'iambic pentameter': '0101010101',
    'trochaic tetrameter': '10101010',
    'trochaic pentameter': '1010101010'
}

#convert a lyric into a list of lists of tuples.
#each tuple contains the word, its phonemes, syllables and cadence
def get_phones_and_syllables(lyrics):
    tokenized_sent = sent_tokenize(lyrics)
    assert len(tokenized_sent) >= 4
    
    musical_lines = []
    for i in range(0, len(tokenized_sent)):
        musical_line = []

        tokens = word_tokenize(tokenized_sent[i])
        tokens.remove(".")
        for word in tokens:
            phonemes = pronouncing.phones_for_word(word)
            if phonemes == []: #if no phonemes we run spellcheck and take best match
                suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=1, include_unknown=True)
                for i in range(0, len(suggestions)):
                    phonemes = pronouncing.phones_for_word(suggestions[i].term)
                    if phonemes:
                        word = suggestions[i].term
                        break

            if phonemes: #If no suitable replacement is found we add original word along with UNK symbols
                tpl = (word, phonemes[0], pronouncing.syllable_count(phonemes[0]), pronouncing.stresses(phonemes[0]))
                musical_line.append(tpl)
            else:
                tpl = (word, 'UNK', -1, 'UNK')
                musical_line.append(tpl)
        
        musical_lines.append(musical_line)

    return musical_lines

#Used to break verse into couplets and quintuplets
def get_sublists(list, n):
    return [list[i:i+n] for i in range(len(list)-n+1)]

#Checks whether or a given pair of tuples contains rhyming words
def check_rhyme(tpl1, tpl2):

    #We can still be sure of rhymes if the unknown words match exactly
    if tpl1[0] == tpl2[0]: return 1
    if tpl1[1] == 'UNK' or tpl2[1] == 'UNK': return 0

    #both words are known and we can check rhymes via the phonemes
    rhyme = 0
    if tpl1[0] in pronouncing.rhymes(tpl2[0]): rhyme = 1
    elif tpl2[0] in pronouncing.rhymes(tpl1[0]): rhyme = 1

    return rhyme

#These functions detect for the presence of a particular 4 line rhyme scheme
def get_rhyme_ABAB(quad_tpl):
    assert len(quad_tpl) == 4, "did not receive list of 4 tuples"
    return check_rhyme(quad_tpl[0], quad_tpl[2]) and check_rhyme(quad_tpl[1],quad_tpl[3])

def get_rhyme_ABBA(quad_tpl):
    assert len(quad_tpl) == 4, "did not receive list of 4 tuples"
    return check_rhyme(quad_tpl[0], quad_tpl[3]) and check_rhyme(quad_tpl[1],quad_tpl[2])

def get_rhyme_AABB(quad_tpl):
    assert len(quad_tpl) == 4, "did not receive list of 4 tuples"
    return check_rhyme(quad_tpl[0], quad_tpl[1]) and check_rhyme(quad_tpl[2],quad_tpl[3])

def get_rhyme_ABAA(quad_tpl):
    assert len(quad_tpl) == 4, "did not receive list of 4 tuples"
    return check_rhyme(quad_tpl[0], quad_tpl[2]) and check_rhyme(quad_tpl[2],quad_tpl[3])

def get_rhyme_AABA(quad_tpl):
    assert len(quad_tpl) == 4, "did not receive list of 4 tuples"
    return check_rhyme(quad_tpl[0], quad_tpl[1]) and check_rhyme(quad_tpl[1],quad_tpl[3])

def get_rhyme_features(musical_data):
    
    last_words = []
    for line in musical_data:
       last_words.append(line[-1]) #we only care about last tuple of each line
    
    #Check frequency of rhyming adjacent couplets
    couplets = get_sublists(last_words, 2)
    couplet_rhymes = []
    for couplet in couplets:
        couplet_rhymes.append(check_rhyme(couplet[0], couplet[1]))
    
    adj_couplet_freq = sum(couplet_rhymes)/len(couplet_rhymes)

    #Check frequency of rhyming spaced couplets
    triplets = get_sublists(last_words, 3)
    spaced_couplet_rhymes = []
    for triplet in triplets:
        spaced_couplet_rhymes.append(check_rhyme(triplet[0], triplet[2]))
    
    spaced_couplet_freq = sum(spaced_couplet_rhymes)/len(spaced_couplet_rhymes)

    #check for presence of 4-line rhyme schemes and dbl space couplets
    abab, abba, aabb, abaa, aaba = 0,0,0,0,0
    quads = get_sublists(last_words, 4)
    dbl_spaced_couplet_rhymes = []

    for quad in quads:
        if get_rhyme_ABAB(quad): abab = 1
        if get_rhyme_ABBA(quad): abba = 1
        if get_rhyme_AABB(quad): aabb = 1
        if get_rhyme_ABAA(quad): abaa = 1
        if get_rhyme_AABA(quad): aaba = 1
        dbl_spaced_couplet_rhymes.append(check_rhyme(quad[0], quad[3]))

    dbl_spaced_freq = sum(dbl_spaced_couplet_rhymes)/len(dbl_spaced_couplet_rhymes)
    
    #Lastly get total freq of pairwise rhymes
    rhyme_pairwise = []
    comb = itertools.combinations(last_words, 2)
    for pair in list(comb): rhyme_pairwise.append(check_rhyme(pair[0], pair[1]))
    pairwise_freq = sum(rhyme_pairwise)/len(rhyme_pairwise)

    return [
        adj_couplet_freq, spaced_couplet_freq, dbl_spaced_freq, 
        abab, abba, aabb, abaa, aaba, pairwise_freq]

def get_rhythmic_features(musical_data, std_meters):

    words = 0
    syllables = 0
    cadences = []

    #build list of cadences for each line
    for line in musical_data:
        cadence = ''
        for tpl in line:
            if not tpl[3] == 'UNK':
                cadence += tpl[3]
                words += 1
                syllables += tpl[2]
        cadences.append(cadence)
    
    word_complexity = (
        words / float(syllables)
        if syllables > 0 else 0)

    #get similarity scores for couplets, spaced and dbl spaced couplets.
    couplets = get_sublists(cadences, 2)
    similarities_couplets = []
    for couplet in couplets: similarities_couplets.append(jaro.jaro_metric(couplet[0], couplet[1]))
    avg_couplet = sum(similarities_couplets)/len(similarities_couplets)
    max_couplet = max(similarities_couplets)

    triplets = get_sublists(cadences, 3)
    similarities_couplets_spaced = []
    for triplet in triplets: similarities_couplets_spaced.append(jaro.jaro_metric(triplet[0], triplet[2]))
    avg_couplet_spaced = sum(similarities_couplets_spaced)/len(similarities_couplets_spaced)
    max_couplet_spaced = max(similarities_couplets_spaced)

    quads = get_sublists(cadences, 4)
    similarities_dbl_spaced = []
    for quad in quads: similarities_dbl_spaced.append(jaro.jaro_metric(quad[0],quad[3]))
    avg_dbl_spaced = sum(similarities_dbl_spaced)/len(similarities_dbl_spaced)
    max_dbl_spaced = max(similarities_dbl_spaced)

    #get avg similarity of all cadences pairwise
    similarities_pairwise = []
    comb = itertools.combinations(cadences, 2)
    for pair in list(comb): similarities_pairwise.append(jaro.jaro_metric(pair[0], pair[1]))

    avg_similarity = sum(similarities_pairwise)/len(similarities_pairwise)

    output = [
            word_complexity, avg_couplet, max_couplet, avg_couplet_spaced, 
            max_couplet_spaced, avg_dbl_spaced, max_dbl_spaced, avg_similarity]

    if not std_meters:
        return output

    std_meter_values = []

    for meter in STANDARD_METERS:
        meter_similarities = []
        for cadence in cadences:
            meter_similarities.append(jaro.jaro_metric(STANDARD_METERS[meter], cadence))
        
        sim_avg = sum(meter_similarities)/len(meter_similarities)
        sim_max = max(meter_similarities)

        std_meter_values.extend([sim_avg, sim_max])

    return output + std_meter_values
        
#This should be the only function to be called for the model.
#It will return a feature vector of all of the desired musical features.
#Note that if rhythmic features are not enabled then std_meters will also not be obtained.
def get_musical_feature_vector(lyrics, rhyming=True, rhythmic=True, std_meters=True):

    musical_data = get_phones_and_syllables(lyrics)
    
    output = []
    if rhyming: output += get_rhyme_features(musical_data)
    if rhythmic: output += get_rhythmic_features(musical_data, std_meters)
    return output



