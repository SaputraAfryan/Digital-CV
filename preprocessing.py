import re
import json
import swifter
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('popular', quiet=True)




class TextPreparation():
    def __init__(self, data=None):
        self.data = data
        self.slang_dict = self.get_slang()
        self.stop_words = self.get_stops()
        self.stemmer = StemmerFactory().create_stemmer()
        self.lemmatizer = WordNetLemmatizer()


    def get_slang(self):
        with open('assets/datasets/slang_dict.json', 'r') as file:
            data = json.load(file)
        return data
    
    def get_stops(self):
        stop_words = set(stopwords.words("indonesian"))
        custom_stopwords = set(['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 
                            'biar', 'bikin', 'bilang', 'krn', 'nya', 'nih', 'sih', 'si',
                            'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja', 
                            'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                            '&amp', 'p', 'jd', 'nek', 'e', 'yo', 'o', 'np', 'nw', 
                            'https', 'http', 't', 'co', 'moilnyaw', 'agshshsjs','dmp',
                            'dfess','zdy','ahiak','wooooy', 'hhahahahahaha','murmuring',
                            'hhaaddeehhhh','eey','raheuuut','hemmmhh','ahahahahahahaha',
                            'bunggg','ndag','urghhhh','xdd','uehe','ewkwkwk','ltl',
                            'aaaakk','rvlog','cuuy','broooooooooo','woiiiiiii','woooooh',
                            'yeuuu','hahahhaahhaahahhah','dihh','yass','utbk','skskskk',
                            'hahahahahhaha','brrip', 'humn','laaaaah','hikd','wkkwkwkwk',
                            'hhahahaah','hahahahahaah','wwkaokwoakwoak','awkakwkawkak',
                            'weyyy','umphhh','pfttttttt','lsc','nyoih','dhhddgh',
                            'uuiii','tbz','hahhhhh','bhak','tteh','nyingnying','pfff',
                            'aiiiihhh','thg','akakakakak','pgw','bradd','wessss','dgm',
                            'miiin','diihh','ajskahsjajsjajsk','woilah','tars','cokkk',
                            'jiaahhhh','wkwkwkwkwkwkwkwk','wkwkwkkwkw','coeg','upis',
                            'jdksbskebsksj','hahehoh','uwuwuwuwuuw', 'aksjsjsksks','jcm',
                            'ndeeeer','hwhw','ququ','boooook','kwah','nyees','geekbahpodcast',
                            'sumargo','flickmagazine','lacuk','wowoowowowowowoo','fassbender',
                            'aaaaak','uwaaaaa','yaelahhh','djdiidhdhsndndjdijdd','fsai',
                            'rrrrrah','aaaaakkk','eeeuuuuhhh','tenenbaum','hadohh','potus',
                            'monek','arggghhhhhh','aaaaaarghh', 'heheheheheueu','iteotfw',
                            'semsem','tobiojersey','idot','morim','sekkk','ejenan',
                            'afkzkunuqkfkigieowkdbjosk','fvck','wkakakka','terrrrrr',
                            'ajsksksjshsjs','dpmallxxi', 'yourwiuwiu','sksksjsj','peele',
                            'huuuuwwaaaahhhh','hahhart','mbahput','huwaaaaaa','fckd',
                            'mmjd','knet','gwangj','nyahahahaha','aaarhh','',
                            'booos','owo','sedeksegaraga','fxxxing','lolll','auliarani',
                            'ehehehhe','hshhshs','theeeeeeee','iqiyi','weeeeehhhh',
                            'tncfu','crotan','huancuuk', 'boxboxid','mchtsvrtn','decc',
                            'ohmy','jff','hukshuks','htgawm','ckli','huweeee',
                            'aaaaarrgghhh','antongekngok','mnet','faaak','trns'
                            'wsowksksnwk','hotrl','falk','egggghhhhhh','fakkkk',
                            'iftw','aokbab','cuklawwww','poonpiriya','nattawut',
                            'kimcop','wkwkwkwkkwkwkwkwk','ftd','ashkskaksjd',
                            'aaaaahggggg','pimchanok','satsetsatset','swadikap',
                            'bebb','hahahahahh','uwma','dangal','dpo','eyyy',
                            'provokemagazine','bhahahahak','seeeeh','wedewww',
                            'tvxq','ueueu','dinitain','lahyaa','hamberr','cexi',
                            'ngentott','wkowkwo','smlmt','fwb','ngeten','whatthefvck',
                            'diputarbelit','cakmangkok', 'ovrl','siiiihhh','daddt',
                            'paspispus','ueueueu','wkaokqowlaowkok','iwww','rdwnfrmnsyh',
                            'lapett','hddu','lsk','asshskhs','wkakakakak','wataheck',
                            'wabb','kamseeeeeeeuuuu','cooy','tefak','mzk','lautnernyaaa',
                            'uben','sils','ngekngok','foxmoviespremiumhd','pbio','tepuuu',
                            'haihhhh', 'tsom', 'wwkwkwkwkk','samoek','brayy','peroah',
                            'pppfffttt','hhahaa','hailaaa','aghf','ahfakk','jooooo',
                            'elaahhhh','kuhhh','hhhhhhhh','wowkwkwk', 'vyll','hiikkkhhh',
                            'ajskqjskaana','wooooiiiiiii','ahelah','rtya','ngebe','rctv',
                            'cuar','hunham','bocen','semprol','gamonin','upill',
                            'lyv','bamf','arghtt','hahsjdbeocnkenfkdnjsbxockskcisndkdcdknxkdmkwndkdnkandksnsnmdjdnskansjbwjsndjdnsjxndmsn',
                            'pelirit','uwoogh','dofp','deeyach','hhahahaaa','hass','mbts',
                            'akwkwk','omgomg','onggg', 'csrnya','leutualy','bebeto','shayy',
                            'widihh','aghny','skskak','wwkwkwkkw','wqwqw','wadohhh','pubg',
                            'habede','kaaah','fuckk', 'yaksip','watchmenid','akwkwoowk',
                            'eskil','klikfilm','molatv','wokwowko','luurrrr','yasss',
                            'waaaaahhhh','duuuhhhh','wooooowww','qwtwfwtwywgsvqqqqwtwuwisip',
                            'tourent', 'hfa','hshdh','tflix','waduu','duilah','drakorindo',
                            'cuiii','pft','arhdjjdb','yaaw','smzbsmale','mbaaakk','shjjshfj','ahahahhh',
                            'terwow','hajahjhs','boomm','ihiyy','snewen','skksks','shshsjsj',
                            'aksksk','aaaahsgehwnahwhwhwvwghq','hadadahhhhh','nderrrr',
                            'aksjsggs','anjahskahs','woowwwww', 'hhhhhhh','ueueueue','heyyyyy',
                            'sjsiznsksk','bahhh','hahhehhoh','shshsh','wtcb','uwahh','hshshshsh',
                            'terrrr','aaaaaak','yihuy','aaargghhh', 'oyeee','hadeeuh','ftf','dipnya',
                            'woohee','nokhorom','ihhhhh','mincot','wkwkwwkwkwkwk','bruhhhhh',
                            'akwkwow','hheu','pisuh','haaaaaaaa', 'dadadahhh','ealaaahh','asdgjhfjkgdfh',
                            'shittt','bensu','mewgulf','yeeehaaaaaa','woohoooooo','skwk','ajdhsnjdhs',
                            'asfsgahshsh','posin','koks','hueeee', 'kuaya','huwawiwew','laff','booooommm',
                            'kubaper','ngiung','awkawk','wkwkkwwk','asshdhgshsjsk','wkwkwkkwkwwkwk',
                            'blaaah','yawlaa','krktr','starbak', 'wtffff','gtv','oalahh','seeeeehhhhh',
                            'momz','mamirt','xzkx','akahdhffvh','hdhejsjdks','wle','hfft','broou',
                            'coooy','woeee', 'hfttt','sksksksks','djaaaannnnccfc','akwjwj','ooooooo',
                            'hhhfftt','kwkwkkw','cihh','laaaaaaa','areyoulostbebiguuyrl','hdeehhh','chuakss',
                            'ponhub','ovt', 'mbaaaa','oooooom','skjsw','whaha','yeyeyeee','ajdghdkh','cenya',
                            'kkekekkk','coyyyy','hahahhahahaha','ahshwvwhsbwvs','otm','arggghh','ngahaha',
                            'eeeemmmm','ahayyyyy','heyhey','tfc','ckckrt','nmn','kulapongvanich','aitthipat',
                            'bitt','kahh','aaaakkkkk','duuuu','mskwydit','wkkwkwkwkwk', 'tchy','ngkoh',
                            'eyyyyy','fahhhh','dsah','eekin','wodnrhwkxneualo','alaaaaa','ueueeee','cucmeyyy',
                            'akhkh','luwh','citraxxi','hahajja', 'hasshhh','oohalah','beeppp','jbcu','waaaawaw',
                            'ahelahhh','ffuffu','hrrrggghh','xoxooxo','arggg','wkwkkwkwkw','ajkdkdks',
                            'aksksksjdkcj','nsd', 'colimon','knetz','soop','plekkkk','heumm','ngerong',
                            'woylah','hmmzz','ahsjskskfjk','huvt','skfnfikejd','mweehehe','iddkdksk','hsjsks',
                            'masnov','lghdtv','lgbtqvwxyz','aksksnnshdksk','hshs','adoyaai','jkfhegfliagwifgiywqgfyqwgfyqgfyiegf',
                            'waaay','nintik','asdghkkl','hwhwhwh','wkakak','aowkwowk','sjsjsjsjs',
                            'aksjskajwyatsh','muahahahahah','lbhs','wkwkwkwjsjsjssj','wqwqwqwq','ftw',
                            'wiboe','tnk','ekwkwk','hshsh','awowkowk','kdpp','meehhh','ajel',
                            'beehh','mgm','angrok','yaowoohh','akakskks','ppkm','meeks','ngokkkk',
                            'akhsks','whwhwh','kstew','fipm','tgc','wasweswos', 'cretttttt','cinn',
                            'ahahabab','etdahhhh','fck','wkwkwkkwkwkwkwkwkkw','nttd','beugh','xnya',
                            'sksksks','vxkxhsnxndmdjdndjdjdn','wwkw','ngederegdeg','borr',
                            'wadau','sembet','emk','shshshshshshshshshsh','huftttttt','yayayay',
                            'aksowkdoek','jasjuss','alhambra','molotovgirl','dantdm','gituhdwndiabwdhabwfh',
                            'dctv','nyetil', 'eeaao','hshshshs','uhuuy','zzoe','atez','bler','blor','sampis',
                            'jlo','iyeel','wooowwww','arhhhg','heeuuuu','wkkwwk', 'wkakakakakaka','uhuyyy',
                            'skwmwkwmwkw','bjir','wkwkwkwkkwu','wewew','bakekok','bnha','ahik','movimax',
                            'paragonxxi','oowhhh','tsang','lgux', 'ngegarong','lahkok','tcog','wzd','jder','ntond',
                            'ginit','wadidaw','kelulus','klux','uwoh','cokk','jedak','junji', 'ahoy','nyumm',
                            'pfoa','waoww','woee','eok','acau','mcdonalds','flis','bekasinians','aaakkk',
                            'awokwok','faakkk','ueueue', 'woooy','sksks','jbjb','awikwok','slebew','bayona',
                            'nsfw','unru','tnh','kpai','jsb','ptj','snaxx','covid', 'gidik','minsik','gamon',
                            'lgbtq','animovie','tasm','jnt','fastfurious','morbius','zsjl','seeeeee','wwkwk',
                            'disneyyy','ndakik', 'imdbnya','cinecrib','wst','ifwt','cmbyn','menfess','wckd',
                            'swank','brou','fsog','ittipat','nkcthi','tabetai','giur', 'ngl','sksk', 'afuk',
                            'jcw','xtina','kroc','blcu','sksksk','trll','jpf','doss','dcu','bvs','cgv','cilers',
                            'wwy', 'tdkr','mjb', 'nwh','bcu','dceu','kdm','okja','cruella','eternals','insidous',
                            'annabel', 'insidius', 'fiftyshades', 'insidiouos'])
        stop_words.update(custom_stopwords)
        return stop_words

    def preprocess_text(self, text):
        text = re.sub(r'[\n\t]|/mvs/|\s+', ' ', text) # remove multiple space, enter, tab, \n, dan \mvs\
        text = re.sub(r"#\w+|@\w+", "", text) # remove hashtag and mention
        text = re.sub(r'https?:\/\/(?:www\.)?\S+', '', text) # remove URLs
        text = re.sub('[^A-Za-z0-9]+', ' ', text) # remove special character
        text = re.sub(r"\d+", "", text) # remove number
        text = text.lower()
        
        tokens = word_tokenize(text) # tokenizing
        norm = [self.slang_dict[token] if token in self.slang_dict else token for token in tokens] # normalization
        tokens = word_tokenize(' '.join(norm)) # re-tokenize
        tokens = [token for token in tokens if token not in self.stop_words] # stopword removal
        tokens = [self.stemmer.stem(token) for token in tokens] # stem
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens] # lemmatization
        tokens = [token for token in tokens if token != ''] # check empty token

        joined_text = ' '.join(tokens)

        return joined_text
    
    def preprocess_df(self):
        if self.data:
            self.data['reviews'] = self.data['reviews'].swifter.apply(self.preprocess_text)
            return self.data

if __name__ == "__main__":
    pass
