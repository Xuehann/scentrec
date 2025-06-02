import os
import sys
sys.path.append(os.path.abspath('./scentrec'))
from utils import ToolGeneral
pwd = os.path.dirname(os.path.abspath('./scentrec'))
tool = ToolGeneral()

class Hyperparams:
    '''Hyper parameters'''
    # Load sentiment dictionary
    deny_word = tool.load_dict(os.path.join(pwd,'scraper','not.txt'))
    posdict = tool.load_dict(os.path.join(pwd,'scraper','positive.txt'))
    negdict = tool.load_dict(os.path.join(pwd,'scraper','negative.txt'))
    pos_neg_dict = posdict|negdict
    # Load adverb dictionary
    mostdict = tool.load_dict(os.path.join(pwd,'scraper','most.txt'))
    verydict = tool.load_dict(os.path.join(pwd,'scraper','very.txt'))
    moredict = tool.load_dict(os.path.join(pwd,'scraper','more.txt'))
    ishdict = tool.load_dict(os.path.join(pwd,'scraper','ish.txt'))
    insufficientlydict = tool.load_dict(os.path.join(pwd,'scraper','insufficiently.txt'))
    overdict = tool.load_dict(os.path.join(pwd,'scraper','over.txt'))
    inversedict = tool.load_dict(os.path.join(pwd,'scraper','inverse.txt'))