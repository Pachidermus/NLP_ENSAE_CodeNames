import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import itertools
import time
import gensim


with open("cm_wordlist.txt") as f:
    cm_wordlist = f.read().split('\n')


class GuesserWordNet:
    """Guesser using WordNet 
    https://www.nltk.org/howto/wordnet.html
    """  
    def __init__(self, min_sim = 0.2, printed_dist = True):
        """Setup GuesserWordNet details

        Args:
        
            max_dist (float, optional):
                minimal wup-similarity threshold under random choices are made
            printed_dist (bool, optional): 
                prints distances from clue to words on the board 
        """
        self.num = 0
        self.min_sim = min_sim
        self.printed_dist = printed_dist
        self.stop_guessing = False

    def set_board(self, words):
        """function that sets words on the board
        """
        self.words = np.array(words)

    def keep_guessing(self):
        """function that defines if guesser keeps guessing
        """
        if self.stop_guessing:
            return False
        else:
            return self.num > 0
        
    def set_clue(self, clue, num):
        """function that sets clue (word and number of words to guess)
        """
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num)
        clue_num = [clue, num]
        return clue_num
    
    def get_answer(self):
        """function that returns a guessed word
        """
        
        self.stop_guessing = False
        sorted_words, sorted_similarities = self._compute_similarity(self.clue, self.words)
        if self.printed_dist: 
            print(f'guesses: {(sorted_words, sorted_similarities)}')
        self.num -= 1
        
        if sorted_similarities[0] < self.min_sim:
            self.stop_guessing = True
            result = np.random.choice([word for word in self.words if word[0]!='*'])
            print(f'The final guess is: {result}')
            return result
        
        elif self.num < 0:
            return None
        
        else :
            result = sorted_words[0]
            print(f'The final guess is: {result}')
            return result

    def _compute_similarity(self, clue, board):
        """function that computes maximum wup-similarities between clue and words from the board
        """
        
        clue_synsets = wn.synsets(clue)
        board_synsets = [wn.synsets(x) for x in board]
        
        similarities = np.zeros(len(board_synsets))
        
        for clue_synset in clue_synsets:
            for word_num, word_synsets in enumerate(board_synsets):
                for word_synset in word_synsets:
                    similarity = clue_synset.wup_similarity(word_synset)
                    if similarity is not None:
                        if similarity > similarities[word_num]:
                            similarities[word_num] = similarity
        

        guesses_order = np.argsort(similarities)[::-1]
        guesses = board[guesses_order] 
        similarities = similarities[guesses_order] 
        
        return guesses, similarities



class CodemasterWordNet_WUP:
    """Codemaster using WordNet 
    https://www.nltk.org/howto/wordnet.html
    This version uses wup-similarity to compute similarities among words and give clues.
    """  
    def __init__(self, cm_wordlist = cm_wordlist, subsample = 0.5, red_team = True, greed = 0.15):
        """Setup CodemasterWordNet_WUP details

        Args:
            cm_wordlist (list of str, optional):
                List of words used as possible clues
            subsample (float, optional):
                Float between 0 and 1 that gives the size of the subsampled vocabulary from cm_wordpool
            red_team (bool, optional): 
                True if Codemaster belongs to the red team, False if Codemaster belongs to the blue team
            greed (float, optional):
                greed parameter that defines how many words the codemaster will make his team guess 
                the higher the greed is, the more words he will make guess
        """
        self.red_team = red_team
        if red_team:
            self.type_player = "Red"
        else:
            self.type_player = "Blue"
        self.cm_wordlist = cm_wordlist
        self.subsample = subsample
        self.greed = greed
        
        self.first_turn = True

    def set_game_state(self, words, maps):
        """function that defines the state of the game : board and map
        """
        self.words = words
        self.maps = maps

    def get_clue(self, nb_words_guess = None):
        """function that returns best clue and number of words to be guessed.
           Returns a tuple (clue word, number of words to guess).
        """
        # map
        team_words = []
        civilian_words=[]
        ennemy_words=[]
        assassin_words=[]
        # Creates Labeled Word arrays separated by types
        for i in range(len(self.words)):       
            if self.words[i][0] == '*':
                continue
            elif self.maps[i] == "Assassin" :
                assassin_words.append(self.words[i].lower())
            elif self.maps[i] == self.type_player :
                team_words.append(self.words[i].lower())
            elif self.maps[i] == "Civilian" :
                civilian_words.append(self.words[i].lower())
            else:
                ennemy_words.append(self.words[i].lower())
                
        if self.red_team :
            print("RED WORDS:\t", team_words)
            print("BLUE WORDS:\t", ennemy_words)
        else :
            print("RED WORDS:\t", ennemy_words)
            print("BLUE WORDS:\t", team_words)
        print("CIVILIAN WORDS:\t", civilian_words)
        print("ASSASSIN WORDS:\t", assassin_words)
        print("\n")
        
        self.nb_words_guess = nb_words_guess
        good_words = team_words
        bad_words = ennemy_words + civilian_words + assassin_words
        
        #If this is the first turn of the algorithm, clean vocabulary to avoid giving unvalid clues
        if self.first_turn:
            #lemmatizer
            st = WordNetLemmatizer()
            st2 = LancasterStemmer()
            #####################
            cm_wordlist = self.cm_wordlist
            for word in self.words :
                # lowercase pb : remove board words with identical lowercase from cm wordlist
                cm_wordlist_lower = [i.lower() for i in cm_wordlist]
                wordlower = word.lower()
                word_in_cm_nocases = list(set([cm_wordlist[i] for i in range(0,len(cm_wordlist)) if cm_wordlist_lower[i]==wordlower]))
                if word_in_cm_nocases != [] :
                    for w in word_in_cm_nocases :
                        cm_wordlist.remove(w)
                # lemmatize or stem to remove words which have same stem/lemma
                word_stem = st.lemmatize(word.lower())
                word_stem2 = st2.stem(word.lower())
                l_remove_stem = []
                for w in cm_wordlist :
                    if (word_stem in (st.lemmatize(w.lower()),st2.stem(w.lower()))) | (word_stem2 in (st.lemmatize(w.lower()),st2.stem(w.lower()))) :
                        l_remove_stem.append(w)
                for i in l_remove_stem :
                    cm_wordlist.remove(i)  
            # final cm wordlist
            self.cm_wordlist = cm_wordlist
            #####################
            self.first_turn = False
        
        #Only use a subsample of the vocabulary at each step to limit computations
        if self.subsample < 1:
            self.cm_wordpool = np.random.choice(self.cm_wordlist, int(len(self.cm_wordlist) * self.subsample), replace=False).tolist()
            
        else:
            self.cm_wordpool = self.cm_wordlist
        
        good_words_size = len(good_words)
        bad_words_size = len(bad_words)
        cm_wordpool_size = len(self.cm_wordpool)
        
        good_words_sim = np.zeros((good_words_size, cm_wordpool_size))
        bad_words_sim = np.zeros((bad_words_size, cm_wordpool_size))
        
        for i, good_word in enumerate(good_words):
            for j, word in enumerate(self.cm_wordpool):
                good_words_sim[i,j] = self._compute_similarity(good_word, word)
                
        for i, bad_word in enumerate(bad_words):
            for j, word in enumerate(self.cm_wordpool):
                bad_words_sim[i,j] = self._compute_similarity(bad_word, word)
        
        #Handle cases where there is only one team word remaining by giving an hypernym or an hyponym
        if len(good_words) <= 1:
            
            no_answer = True
            good_syns = wn.synsets(good_words[0])
            
            for good_syn in good_syns:
                
                hypernyms = good_syn.hypernyms()
                
                if len(hypernyms) > 0:
                    
                    node = hypernyms[0]
                    clue = self._get_string_from_node(node)
                    no_answer = False
                    break
                    
                hyponyms = good_syn.hyponyms()
                
                if len(hyponyms) > 0:
                    
                    node = hyponyms[0]
                    clue = self._get_string_from_node(node)
                    no_answer = False
                    break
            
            if no_answer:
                
                clue = np.random.choice(self.cm_wordpool)
            
            return clue, len(good_words)
        
        #if there are at least two team words remaining, compute best combinations
        else:
        
            if nb_words_guess is not None:

                best_word_num, best_combination_num, _ = self._compute_best_move(good_words_sim, bad_words_sim, nb_words_guess)

                final_clue = (self.cm_wordpool[best_word_num], nb_words_guess)
                final_combo = np.array(good_words)[[best_combination_num]]

            else:

                best_words_num = list()
                best_combinations_num = list()
                scores = list()
                nums = list()
                
                for num in np.arange(1, 5):

                    best_word_num, best_combination_num, score = self._compute_best_move(good_words_sim, bad_words_sim, num)
                    best_words_num.append(best_word_num)
                    best_combinations_num.append(best_combination_num)
                    scores.append(score)
                    nums.append(num)
                
                greed_vector = np.sqrt(np.arange(len(scores)))
                greed_vector[0] -= self.greed / 2
                best_score_arg = np.argmax(np.array(scores) + self.greed * greed_vector)
                final_clue = (self.cm_wordpool[best_words_num[best_score_arg]], nums[best_score_arg])
                final_combo = np.array(good_words)[[best_combinations_num[best_score_arg]]]
            
            print(f"Clue : {final_clue}")
            print(f"Combo : {final_combo}")

            return final_clue
    
    def _compute_similarity(self, word1, word2):
        """function that computes similarity between two words given as strings.
        """
        word1_synsets = wn.synsets(word1)
        word2_synsets = wn.synsets(word2)
        
        result = 0
        
        for synset1 in word1_synsets:
            for synset2 in word2_synsets:
                r = synset1.wup_similarity(synset2)
                if r is not None:
                    if r > result:
                        result = r
        return result
    
    def _compute_best_move(self, good_words_sim, bad_words_sim, num):
        
        good_words_size = good_words_sim.shape[0]
        cm_wordpool_size = len(self.cm_wordpool)
        
        combination_sim_min = 0
        
        best_word_num = 0
        associated_combination_num = 0
        
        for combination in itertools.combinations(np.arange(good_words_size),num):
            for word in np.arange(cm_wordpool_size):
                max_bad_sim = np.max(bad_words_sim[:, word])
                current_combination_sim_min = np.min(good_words_sim[tuple(combination), word])
                if (current_combination_sim_min > combination_sim_min) & (current_combination_sim_min > max_bad_sim):
                    combination_sim_min = current_combination_sim_min
                    best_word_num = word
                    associated_combination_num = combination
                
        return best_word_num, associated_combination_num, combination_sim_min

    def _get_string_from_node(self, node):
        """function that get a clean string from a synset node.
        """
        node_strings = node.lemma_names()
        words = [word.lower() for word in self.words]
        
        for i, node_string in enumerate(node_strings):
            
            if '_' in node_string or '-' in node_string:
                
                candidate = node_string.split('_')[-1].split('-')[-1]
                
                if candidate.upper() in words:
                    
                    candidate = node_string.split('_')[0].split('-')[0]
            
                node_strings[i] = candidate
                
            if node_strings[i].upper() not in words:
                
                return node_strings[i]
        
        return 'random'

            
            
class CodemasterWordNet_HYP:
    """Codemaster using WordNet 
    https://www.nltk.org/howto/wordnet.html
    This version uses lowest common hypernyms to choose clues.
    """ 
    def __init__(self, red_team = True, greed = -0.05):
        """Setup CodemasterWordNet_HYP details

        Args:
            red_team (bool, optional): 
                True if Codemaster belongs to the red team, False if Codemaster belongs to the blue team
            greed (float, optional):
                greed parameter that defines how many words the codemaster will make his team guess 
                the higher the greed is, the more words he will make guess
        """
        self.red_team = red_team
        if red_team:
            self.type_player = "Red"
        else:
            self.type_player = "Blue"
        self.greed = greed

    def set_game_state(self, words, maps):
        """function that defines the state of the game : board and map
        """
        self.words = words
        self.maps = maps

    def get_clue(self, nb_words_guess = None):
        """function that returns best clue and number of words to be guessed.
           Returns a tuple (clue word, number of words to guess).
        """
        # map
        team_words = []
        civilian_words=[]
        ennemy_words=[]
        assassin_words=[]
        # Creates Labeled Word arrays separated by types
        for i in range(len(self.words)):       
            if self.words[i][0] == '*':
                continue
            elif self.maps[i] == "Assassin" :
                assassin_words.append(self.words[i].lower())
            elif self.maps[i] == self.type_player :
                team_words.append(self.words[i].lower())
            elif self.maps[i] == "Civilian" :
                civilian_words.append(self.words[i].lower())
            else:
                ennemy_words.append(self.words[i].lower())
                
        if self.red_team :
            print("RED WORDS:\t", team_words)
            print("BLUE WORDS:\t", ennemy_words)
        else :
            print("RED WORDS:\t", ennemy_words)
            print("BLUE WORDS:\t", team_words)
        print("CIVILIAN WORDS:\t", civilian_words)
        print("ASSASSIN WORDS:\t", assassin_words)
        print("\n")
        
        self.nb_words_guess = nb_words_guess
        good_words = team_words
        bad_words = ennemy_words + civilian_words + assassin_words
        
        #Handle cases where there is only one team word remaining by giving an hypernym or an hyponym
        if len(good_words) <= 1:
            
            no_answer = True
            good_syns = wn.synsets(good_words[0])
            
            for good_syn in good_syns:
                
                hypernyms = good_syn.hypernyms()
                
                if len(hypernyms) > 0:
                    
                    node = hypernyms[0]
                    clue = self._get_string_from_node(node)
                    no_answer = False
                    break
                    
                hyponyms = good_syn.hyponyms()
                
                if len(hyponyms) > 0:
                    
                    node = hyponyms[0]
                    clue = self._get_string_from_node(node)
                    no_answer = False
                    break
            
            if no_answer:
                
                clue = 'random'
            
            return clue, len(good_words)
        
        #If there are at least two team words remaining, compute all 2-words combinations and select best
        else:
            
            combination_words = []
            combination_similarities = []
            combination_args = []

            for combination in itertools.combinations(good_words, 2):
                
                combination_words.append(combination)
                max_sim, max_syn_arg1, max_syn_arg2 = self._compute_similarity(combination[0], combination[1])
                combination_similarities.append(max_sim)
                combination_args.append([max_syn_arg1, max_syn_arg2])

            combination_max_sim = np.argmax(combination_similarities)
            combo = list(combination_words[combination_max_sim])
            combo_args = combination_args[combination_max_sim]
            combo1 = wn.synsets(combo[0])[combo_args[0]]
            combo2 = wn.synsets(combo[1])[combo_args[1]]
            node = combo1.lowest_common_hypernyms(combo2)
            
            if len(node) > 0:
                node = node[0]
            
            else:
                node = combo1.hypernyms()
                
                if len(node) > 0:
                    node = node[0]
                    
                else:
                    node = combo1.hyponyms()[0]

            good_words = set(good_words) - set(combo)
               
            #Seek if there are better combinations with 3 or 4 words (if enough words are remaining).
            for num in range(2,5):
                
                if len(good_words) == 0:
                    break
                
                combination_words = []
                combination_similarities = []
                combination_args = []

                for good_word in good_words:
                    combination_words.append(good_word)
                    max_sim, max_syn_arg = self._compute_similarity_with_ref(node, good_word)
                    combination_similarities.append(max_sim)
                    combination_args.append(max_syn_arg)
                
                combination_similarities_bad = []
                
                for bad_word in bad_words:
                    max_sim, max_syn_arg = self._compute_similarity_with_ref(node, bad_word)
                    combination_similarities_bad.append(max_sim)
                
                if np.max(combination_similarities_bad) > np.max(combination_similarities) + self.greed:
                    break

                combination_max_sim = np.argmax(combination_similarities)    
                combo.append(combination_words[combination_max_sim])
                combo_arg = combination_args[combination_max_sim]
                best_synset = wn.synsets(combo[-1])[combo_arg]
                node = node.lowest_common_hypernyms(best_synset)[0]
                
                good_words = set(good_words) - set(combo)

            clue = self._get_string_from_node(node)

            print(f"Clue : {clue} {len(combo)}")
            print(f"Combo : {combo}")
            
            return clue, len(combo)
    
    def _compute_similarity(self, word1, word2):
        """function that computes similarity between two words given as strings.
        """
        word1_synsets = wn.synsets(word1)
        word2_synsets = wn.synsets(word2)
        
        result = 0
        closest_syn1 = None
        closest_syn2 = None
        
        for i, synset1 in enumerate(word1_synsets):
            for j, synset2 in enumerate(word2_synsets):
                r = synset1.wup_similarity(synset2)
                if r is not None:
                    if r > result:
                        result = r
                        closest_syn1 = i
                        closest_syn2 = j
        return result, closest_syn1, closest_syn2
    
    def _compute_similarity_with_ref(self, synset_ref, word):
        """function that computes similarity between a given synset and a word given as a string.
        """
        word_synsets = wn.synsets(word)
        
        result = 0
        closest_syn = None
        
        for i, synset in enumerate(word_synsets):
            r = synset_ref.wup_similarity(synset)
            if r is not None:
                if r > result:
                    result = r
                    closest_syn = i
        return result, closest_syn

    def _get_string_from_node(self, node):
        """function that get a clean string from a synset node.
        """
        node_strings = node.lemma_names()
        words = [word.lower() for word in self.words]
        
        for i, node_string in enumerate(node_strings):
            
            if '_' in node_string or '-' in node_string:
                
                candidate = node_string.split('_')[-1].split('-')[-1]
                
                if candidate.upper() in words:
                    
                    candidate = node_string.split('_')[0].split('-')[0]
            
                node_strings[i] = candidate
                
            if node_strings[i].upper() not in words:
                
                return node_strings[i]
        
        return 'random'
        