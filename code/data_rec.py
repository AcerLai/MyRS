import json
import numpy as np

def data_pro(route_json, route_data, dnn=False):
    f_format = open(route_json, "r+")
    f_format = json.load(f_format)
    f_format = f_format['inputs']
    f_format[-6]['share_dict'] = 'gametype'
    gametype_share, aid_share, prov_share, city_share = [5, 7, 8], [3, 6], [0, 4], [1, 2]

    f = open(route_data, "r+")
    lines = f.readlines()
    if dnn:
        label, label_weight, feature = [], [], [[] for i in range(16)]
    else:
        label, label_weight, feature = [], [], [[] for i in range(24)]
    index = 1
    
    for line in lines:
        tmp = 0
        gametype_feature, aid_feature, prov_feature, city_feature = [], [], [], []
        a_dist_feature, u_dist_feature, a_conts_feature, u_conts_feature = [], [], [], []
        gametype_weight, aid_weight, prov_weight, city_weight = [], [], [], []
        a_dist_weight, u_dist_weight = [], []
        labels, features = line.split('|')
        features = features.split(';')
        labels, label_weights = labels.split(':')
        labels = int(labels)
        label_weights = float(label_weights)
        label.append(labels)
        label_weight.append(label_weights)

        for i in prov_share:
            fea = features[i].split(',')
            for j in fea:
                v, w = j.split(':')
                prov_feature = prov_feature + [int(v)]
                prov_weight = prov_weight + [float(w)]
        feature[tmp].append(prov_feature)
        tmp += 1 
        feature[tmp].append(prov_weight)
        tmp += 1

        for i in city_share:
            fea = features[i].split(',')
            for j in fea:
                v, w = j.split(':')
                city_feature = city_feature + [int(v)]
                city_weight = city_weight + [float(w)]
        feature[tmp].append(city_feature)
        tmp += 1
        feature[tmp].append(city_weight)
        tmp += 1
        
        for i in aid_share:
            if not dnn:
                aid_feature, aid_weight = [], []
            fea = features[i].split(',')
            l = len(fea)
            for j in fea:
                v, w = j.split(':')
                aid_feature = aid_feature + [int(v)]
                aid_weight = aid_weight + [float(w)]
            aid_feature = aid_feature + [0.0] * (30 - l)
            aid_weight = aid_weight + [0.0] * (30 - l)
            
            if not dnn:
                feature[tmp].append(aid_feature)
                tmp += 1
                feature[tmp].append(aid_weight)
                tmp += 1
        
        if dnn:
            feature[tmp].append(aid_feature)
            tmp += 1
            feature[tmp].append(aid_weight)
            tmp += 1

        for i in gametype_share:
            if not dnn:
                gametype_feature, gametype_weight = [], []
            fea = features[i].split(',')
            l = len(fea)
            for j in fea:
                v, w = j.split(':')
                gametype_feature = gametype_feature + [int(v)]
                gametype_weight = gametype_weight + [float(w)]
            gametype_feature = gametype_feature + [0.0] * (30 - l)
            gametype_weight = gametype_weight + [0.0] * (30 - l)
                    
            if not dnn:
                feature[tmp].append(gametype_feature)
                tmp += 1
                feature[tmp].append(gametype_weight)
                tmp += 1
        if dnn:     
            fea = features[-6].split(',')
            v, w = j.split(':')
            gametype_feature = gametype_feature + [int(v)]
            gametype_weight = gametype_weight + [float(w)]
            feature[tmp].append(gametype_feature)
            tmp += 1
            feature[tmp].append(gametype_weight)
            tmp += 1
            


        fea = features[-2].split(',')
        for j in fea:
            v, w = j.split(':')
            a_dist_feature = a_dist_feature + [int(v)]
            a_dist_weight = a_dist_weight + [float(w)]
        feature[tmp].append(a_dist_feature)
        tmp += 1
        feature[tmp].append(a_dist_weight)
        tmp += 1

        fea = features[-1].split(',')
        for j in fea:
            v, w = j.split(':')
            u_dist_feature = u_dist_feature + [int(v)]
            u_dist_weight = u_dist_weight + [float(w)]
        feature[tmp].append(u_dist_feature)
        tmp += 1
        feature[tmp].append(u_dist_weight)
        tmp += 1

        fea = features[-5]
        v, w = fea.split(':')
        target = [int(v)]
        target_weight = [float(w)]
        feature[tmp].append(target)
        tmp += 1
        feature[tmp].append(target_weight)
        tmp += 1
        
        if not dnn:
            fea = features[-6]
            v, w = fea.split(':')
            target_gametype = [int(v)]
            target_gametype_weight = [float(w)]
            feature[tmp].append(target_gametype)
            tmp += 1
            feature[tmp].append(target_gametype_weight)
            tmp += 1
        
        fea = features[-4].split(',')
        for j in fea:
            a_conts_feature = a_conts_feature + [float(j)]
        feature[tmp].append(a_conts_feature)
        tmp += 1
        
        fea = features[-3].split(',')
        for j in fea:
            u_conts_feature = u_conts_feature + [float(j)]
        feature[tmp].append(u_conts_feature)

        if index % 100000 == 0:
            print(index)
        index += 1
        
    feature = [np.array(i) for i in feature]
        
    return feature, np.array(label), np.array(label_weight)








