import visualization as vs
import pickle

f = open(r'./trainLog/GenLog_8x8_trans_True','rb')
GenLog = (pickle.load(f)+1)/2
vs.CV2_GENLOG_SHOW(GenLog,1,10,5)