#Hosein Seifi حسین سیفی
#810100386
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
wiki = [f"wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
gutenberg = ["pg16457.txt"]
bpe_trainer = BpeTrainer(vocab_size = 1000000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
wp_trainer = WordPieceTrainer(vocab_size = 1000000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
f = open(gutenberg[0], "r")
g = f.read()
f.close()
del f
#%%
BPE_gutenberg = Tokenizer(BPE(unk_token="[UNK]"))
BPE_gutenberg.pre_tokenizer = Whitespace()
BPE_gutenberg.train(gutenberg, bpe_trainer)
#%%
BPE_wiki = Tokenizer(BPE(unk_token="[UNK]"))
BPE_wiki.pre_tokenizer = Whitespace()
BPE_wiki.train(wiki, bpe_trainer)
#%%
wp_gutenberg = Tokenizer(WordPiece(unk_token="[UNK]"))
wp_gutenberg.pre_tokenizer = Whitespace()
wp_gutenberg.train(gutenberg, wp_trainer)
#%%
wp_wiki = Tokenizer(WordPiece(unk_token="[UNK]"))
wp_wiki.pre_tokenizer = Whitespace()
wp_wiki.train(wiki, wp_trainer)
#%%
bpe_gutenberg_tokens = BPE_gutenberg.encode(g).tokens
wp_gutenberg_tokens = wp_gutenberg.encode(g).tokens
bpe_wiki_tokens = BPE_wiki.encode(g).tokens
wp_wiki_tokens = wp_wiki.encode(g).tokens
#%%