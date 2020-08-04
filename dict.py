from collections import OrderedDict

od1 = OrderedDict()
od1['accouracy']=list()
od1['average_time']=list()
od1['total_time']=list()

od2 = OrderedDict()
od2['accouracy']=list()
od2['radius_time']=list()
od2['average_time']=list()
od2['total_time']=list()

od3 = OrderedDict()
od3['accouracy']=list()
od3['membership_time']=list()
od3['average_time']=list()
od3['total_time']=list()


algorithm_list = [
                    {
                     'label':'KNN',
                     'data': od1, 
                     'average_result_accouracy':0,
                     'average_result_time':0,
                     'best_result_accouracy':0,
                     'best_result_time':0
                    },
                    {
                     'label':'AdaKNN',
                     'data': od2,
                     'average_result_accouracy':0,
                     'average_result_time':0,
                     'best_result_accouracy':0,
                     'best_result_time':0
                    },
                    {
                     'label':'FKNN',
                     'data': od1,
                     'average_result_accouracy':0,
                     'average_result_time':0,
                     'best_result_accouracy':0,
                     'best_result_time':0
                    },
                    {
                     'label':'It2FKNN',
                     'data': od3,
                     'average_result_accouracy':0,
                     'average_result_time':0,
                     'best_result_accouracy':0,
                     'best_result_time':0
                    },
                    {
                     'label':'FRNN',
                     'data': od1,
                     'average_result_accouracy':0,
                     'average_result_time':0,
                     'best_result_accouracy':0,
                     'best_result_time':0
                    },
                    {
                     'label':'FRNN FRS',
                     'data': od1,
                     'average_result_accouracy':0,
                     'average_result_time':0,
                     'best_result_accouracy':0,
                     'best_result_time':0
                    },
                    {
                     'label':'FRNN VQRS',
                     'data': od1,
                     'average_result_accouracy':0,
                     'average_result_time':0,
                     'best_result_accouracy':0,
                     'best_result_time':0
                    }
                ]