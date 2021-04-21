import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import itertools
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import random


class GuesserGlove:
    """Guesser using Glove embedding
    """   
    def __init__(self, model=None, printed_dist=True, max_dist=1):
        """Setup GuesserGlove details

        Args:
        
            model (:class:`gensim.models.keyedvectors.Word2VecKeyedVectors`):
                GloVe model
            printed_dist (bool, optional): 
                prints distances from clue to words on the board 
            max_dist (float, optional):
                threshold distance above which words will not be guessed
        """
        self.model = model
        self.printed_dist = printed_dist
        self.max_dist = max_dist
        
        self.model_vecs = model.vocab
        self.num = 0

    def set_board(self, words):
        """function that sets words on the board
        """
        self.words = words

    def keep_guessing(self):
        """function that defines if guesser keeps guessing
        """
        return self.num > 0
    
    def set_clue(self, clue, num):
        """function that sets clue (word and number of words)
        """
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num)
        li = [clue, num]
        return li

    def get_answer(self):
        """function that returns a guessed word
        """
        sorted_words = self._compute_distance(self.clue, self.words)
        if self.printed_dist == True :
            print(f'guesses: {[(w, round(d,3)) for d,w in sorted_words]}')
        self.num -= 1
        
        # print word if its distance is inferior to max_dist
        try :
            if sorted_words[0][0] > self.max_dist :
                return None
            else :
                print("The final guess is:", sorted_words[0][1])
                return sorted_words[0][1]
        except :
            random.seed(266)
            remaining_words = [word for word in self.words if "*" not in word]
            random_choice = random.choice(range(len(remaining_words)))
            print("The final guess is:", remaining_words[random_choice])
            return remaining_words[random_choice]
            
    def _compute_distance(self, clue, board):
        """function that computes distance between clue and words from the board
        """
         #  list all vocabulary words from glove embedding
        vocab = list(self.model_vecs)   
        glove = []
        for word in board:
            try:
                if word[0] == '*':
                    continue
                # use cosine distance between each word of the board 
                glove.append((self.model.distance(clue,word.lower()),word))    
            except KeyError:
                continue
        glove = list(sorted(glove))
        return glove
    
    
class CodemasterGlove:
    """Codemaster using Glove embedding
    """  
    def __init__(self, model=None, red_team = True, nb_words_guess=None, vocab_size = 1000, team_w = 3, ennemy_w = 1, civilian_w = 1, assassin_w = 1, printed_dist=True, greed=0.1):
        """Setup CodemasterGlove details

        Args:
        
            model (:class:`gensim.models.keyedvectors.Word2VecKeyedVectors`):
                GloVe model
            red_team (bool, optional): 
                True if Codemaster belongs to the red team, False if Codemaster belongs to the blue team
            nb_words_guess (int from 1 to 3, optional): 
                number of words the codemaster want to make the player guess. If None, the codemaster will choose the
                best number of words to be guessed according to a greed threshold
            vocab size (int, optional): 
                size of codemaster's vocabulary
            team_w (float, optional):
                weight for team's words
            ennemy_w (float, optional):
                weight for other team's words
            civilian_w (float, optional):
                weight for civilian's words
            assassin_w (float, optional): 
                weight for assassin's words
            printed_dist (bool, optional):
                prints distances from chosen clue words from words on the board
            greed (float, optional):
                greed parameter that defines how many words the codemaster will make his team guess. 
                the higher the greed is, the more words he will make guess
        """
        self.model = model
        if red_team:
            self.type_player = "Red"
        else:
            self.type_player = "Blue"
        self.nb_words_guess = nb_words_guess
        self.vocab_size = vocab_size
        self.team_w = team_w
        self.ennemy_w = ennemy_w
        self.civilian_w = civilian_w
        self.assassin_w = assassin_w
        self.printed_dist=printed_dist
        self.greed=greed
        
        self.model_vecs = model.vocab
        self.cm_wordlist=None
        self.bad_word_dists = None
        self.team_word_dists = None
        

    def set_game_state(self, words, maps):
        """function that defines the state of the game : board and map
        """
        self.words = words
        self.maps = maps

    def compute_distances(self,group_wordlist,cm_wordlist) :
        """function that computes distances between words from group_wordlist and words from cm_wordlist.
           Returns a dictionary containing distances
           
       Args:

           group_wordlist (list): 
               list of words from the board, for a given group (red team, blue team, assassin, civilians)
           cm_wordlist (list): 
               codemaster's vocabulary
        """
        group_word_dists = {}
        
        vocab = list(self.model_vecs)
        for word in group_wordlist:
            group_word_dists[word] = {}

            # lowercase word, capitalize it if not in vocab
            word2 = word.lower()
            if word2 not in vocab :
                word2 = word2.capitalize()

            # iterate for all val in cm_wordlist
            for val in cm_wordlist :
                b_dist = self.model.distance(val, word2)
                group_word_dists[word][val] = b_dist   
                
        return group_word_dists
    
        
    def get_clue(self):
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
                
        if self.type_player == "Red" :
            print("RED WORDS:\t", team_words)
            print("BLUE WORDS:\t", ennemy_words)
        else :
            print("RED WORDS:\t", ennemy_words)
            print("BLUE WORDS:\t", team_words)
        print("CIVILIAN WORDS:\t", civilian_words)
        print("ASSASSIN WORDS:\t", assassin_words)
        print("\n")

        # vocab from embedding glove
        vocab = list(self.model_vecs)
        
        #lemmatizer
        st = WordNetLemmatizer()
        st2 = LancasterStemmer()
        
        #### cm_wordlist : take most common words from glove embedding vocab
        most_common_vocab = dict()
        for word in self.model_vecs:
            most_common_vocab[word] = self.model_vecs[word].count
        cm_wordlist = sorted(most_common_vocab, key=most_common_vocab.get, reverse=True)[:self.vocab_size]
        for word in self.words :
            # remove from cm_wordlist words which are identical or have the same stem/lemma 
            word_stem = st.lemmatize(word.lower())
            word_stem2 = st2.stem(word.lower())
            l_remove_stem=[]
            for w in cm_wordlist :
                if (word_stem in (st.lemmatize(w.lower()),st2.stem(w.lower()))) | (word_stem2 in (st.lemmatize(w.lower()),st2.stem(w.lower()))) :
                    l_remove_stem.append(w)
            cm_wordlist = [word for word in cm_wordlist if word not in l_remove_stem] 
        # final cm wordlist
        self.cm_wordlist=cm_wordlist
        
        # compute distance between words from the board and words from cm_wordlist (use cosine distance)
        if not self.bad_word_dists:
            self.team_word_dists = self.compute_distances(team_words,self.cm_wordlist)
            self.civilian_word_dists = self.compute_distances(civilian_words,self.cm_wordlist)
            self.assassin_word_dists = self.compute_distances(assassin_words,self.cm_wordlist)
            self.ennemy_word_dists = self.compute_distances(ennemy_words,self.cm_wordlist)

        best_clues = dict()
        best_dists = dict()
        
        # for all possible number of words to guess (1 to 4 here) : find best combo, best distance
        for clue_num in range(1,np.min([len(team_words)+1,5])):
            best_clue = None
            best_dist = np.inf
            guess = None
            # iterate over all possible sets of clue_num number of words 
            for team_word_comb in list(itertools.combinations(team_words, clue_num)):
                l_team_word_comb = list(team_word_comb) # list which contains a combo of words (ex : ["banana","mango"])
                avg_dist_list = dict()
                # iterate for all word in cm wordlist 
                for val in self.cm_wordlist :
                    team_word_comb_dist_val = [self.team_word_dists[word][val] for word in l_team_word_comb]
                    ennemy_word_dist_val = [self.ennemy_word_dists[word][val] for word in ennemy_words]
                    assassin_word_dist_val = [self.assassin_word_dists[word][val] for word in assassin_words]
                    civilian_word_dist_val = [self.civilian_word_dists[word][val] for word in civilian_words]
                        
                    # computes a weighted average distance associated with the specific word from cm wordlist, and specific combo
                    avg_dist_list[val] = np.average([np.mean(team_word_comb_dist_val),
                                                     np.mean([2-i for i in ennemy_word_dist_val]),
                                                     np.mean([2-i for i in assassin_word_dist_val]),
                                                     np.mean([2-i for i in civilian_word_dist_val])
                                                    ], 
                                                    weights = [self.team_w,self.ennemy_w,self.assassin_w,self.civilian_w])
                
                # get best clue for specific combo of clue_num_words, according to average distance
                if min(avg_dist_list.values()) < best_dist :
                    best_clue = min(avg_dist_list, key=avg_dist_list.get)
                    best_dist = min(avg_dist_list.values())
                    best_guess = team_word_comb
            if self.printed_dist == True :
                if len(best_guess) == 1 :
                    print("TRY TO GUESS",clue_num,"WORD:")
                else :
                    print("TRY TO GUESS",clue_num,"WORDS:")
                print("\tBest combination:",best_guess)
                print("\tBest clue for this combo:", (best_clue,round(best_dist,3)))
                    
            # best clues and dist only for best combo of clue_num words to guess
            best_clues[best_guess] = best_clue
            best_dists[best_guess] = best_dist
    
        # using the greed parameter to determine how many words to be guessed :
        if self.nb_words_guess == None :
            len_team_words = len(team_words)
            ratio_ennemy_team = np.min([1,len(ennemy_words)/len(team_words)])
            # guess a certain number of words according to the value of ratio of remaining words ennemy/team
            if ratio_ennemy_team < 0.4 :
                nb_guesses = np.min([len_team_words,4])
            elif ratio_ennemy_team < 0.6 :
                nb_guesses = np.min([len_team_words,3])
            elif ratio_ennemy_team < 0.8 :
                nb_guesses = np.min([len_team_words,2])
            else :
                nb_guesses = np.min([len_team_words,1])
            # update the number of words to guess according to greed parameter and number of words already guessed
            nb_guesses_update = round(nb_guesses+self.greed*5*(len([w for w in self.maps if w==self.type_player])-len_team_words+1))
            nb_guesses=np.min([nb_guesses_update,4,len_team_words])
            print("\nGreed parameter =", self.greed,":")
            best_combo={key: nb_guesses for key in best_dists.keys() if len(key) == nb_guesses}
            final_combo = list(best_combo.keys())[0]
            final_clue = best_clues[final_combo]
            
        # if the codemaster already knew how many words he wanted to make the other players guess
        else :
            if self.nb_words_guess <= len(list(best_clues.keys())) :
                final_combo = list(best_clues.keys())[self.nb_words_guess-1]
            else :
                final_combo = list(best_clues.keys())[-1]
            final_dist = best_dists[final_combo]
            final_clue = best_clues[final_combo]
        print("\n")
        print("FINAL CLUE :", final_clue)
        print("FOR COMBO :", final_combo)
        print("="*100)
        print("GUESSER'S TURN")        
        
        return(final_clue,len(final_combo))