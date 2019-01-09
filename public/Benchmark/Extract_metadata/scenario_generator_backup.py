import sys
from contextlib import closing
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime
import traceback
import os
import errno
from scipy.spatial import distance
from scipy.cluster import hierarchy as hier
from scipy.spatial import distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage
import sklearn.metrics.pairwise
import random
from sklearn.model_selection import KFold

# def semantic_distance(u, v):
# 	return np.mean([distance.euclidean(u[:100],v[:100]), 
# 		              distance.euclidean(u[:100],v[100:]),
# 		              distance.euclidean(u[100:],v[:100]),
# 		              distance.euclidean(u[100:],v[100:])])

def get_node_id(value):
	command = "node_id" + " " + value
	return os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()


def random_walk(spo, maxwalks = 1000, max_depth = 3):

	command = "hpath" + " " + get_node_id(spo[0]) + " " + get_node_id(spo[1]) + " "  +  "NULL" + " " + str(max_depth) + " " + "F" + " " + "P"
	features = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
	features =  [path.split("\n")[1:-1] for path in features]

	# features = [s.rstrip(',').split(',') for s in features]
	random_prob = {}
	for path in features:
		path = path.rstrip(',').split(',')
		random_prob[path] = {}
		cnt = 0
		for walk in xrange(maxwalks):
			node = spo[0]
			for r in path:
				command = "neighborwithrel" + " " + node + " " + r + " " + "TRUE"
				relnbrs = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
				relnbrs =  relnbrs.split('\n')[1:-1]
				n_nbrs = len(relnbrs)
				if n_nbrs == 0:
					break # restart random walk
				else:
					node = np.random.choice(relnbrs, 1)[0] # pick 1 nbr uniformly at random
			if node in random_prob[path]:
				random_prob[path][node] += 1
			else:
				random_prob[path][node] = 1
			# if node == spo[1]:
			# 	cnt += 1
		# print '\t{}. {}/{} for {}'.format(pthidx+1, cnt, maxwalks, path)
	# mat[idx, pthidx] = cnt / float(maxwalks)
	ids =  open("random_walk.tsv", 'w')
	for path in random_prob.keys():
		ids.write(path + " : ")
		for node in random_prob[path].keys():
			ids.write(node + " ("+ str(random_porb[path][node]) + ") " + "\t")
		ids.write("\n")
	ids.close()

def get_overlapping_pair(target_relation, relations):
	overlap_pair = []
	exclude = []
	for similar_relation in relations:
		command = "getoverlappingpairs" + " " + target_relation + " " + similar_relation + " TRUE"
		pairs_ = os.popen("echo \"" + command +"\" | socat -t 3600 -T 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
		pairs_ = pairs_.split('\n')[1:-1]
		if (len(pairs_) < 10):
			exclude.append(similar_relation)
		else:
			pairs_ = [pair.split('\t') for pair in pairs_]	
			pairs_ =[(pair[0], pair[1], target_relation) for pair in pairs_]
			overlap_pair.append(pairs_)
	overlap_pair = sum(overlap_pair, [])
	overlap_pair = list(set(overlap_pair))
	overlap_pair = [list(x) for x in overlap_pair]
	return exclude, overlap_pair

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
	parser.add_argument('basedir', metavar='base', help='experiment base directory')
	parser.add_argument('-r', type=str, required=True,
            dest='relation', help='Relation')
	parser.add_argument('-rx', nargs='+', required=True, 
			dest='similar_relation', help='Similar Relation')
	parser.add_argument('-lv', type=str, required=True,
            dest='level', help='Difficulty of level')
	parser.add_argument('-hm', type=float, required=True,
            dest='homo', help='Homogeneity')
	parser.add_argument('-cs', type=float, required=True,
            dest='consis', help='Consistency')
	parser.add_argument('-sc', type=int, required=True,
            dest='size_scenario', help='Size of Scenario')
	parser.add_argument('-pn', type=float, required=True,
            dest='pos_neg', help='Ratio of positive/negative')
	args = parser.parse_args()
   
	if not os.path.isfile(args.basedir + "splits/" + args.relation + "/" + args.relation + ".tsv"):

		file1_names = open(args.basedir + "relation_sets/labeled_edges.tsv", 'r')

		if not os.path.exists(os.path.dirname(args.basedir + "splits/" + args.relation + "/" + args.relation + ".tsv")):
			try:
				os.makedirs(os.path.dirname(args.basedir + "splits/" + args.relation + "/" + args.relation + ".tsv"))
			except OSError as exc: # Guard against race condition:
				if exc.errno != errno.EEXIST:
					raise

		if not os.path.isfile(args.basedir + "splits/" + args.relation + "/" + args.relation + "_id.tsv"):
			if not os.path.exists(os.path.dirname(args.basedir + "splits/" + args.relation + "/" + args.relation + "_id.tsv")):
				try:
					os.makedirs(os.path.dirname(args.basedir + "splits/" + args.relation + "/" + args.relation + "_id.tsv"))
				except OSError as exc: # Guard against race condition
					if exc.errno != errno.EEXIST:
						raise

		ids =  open(args.basedir + "splits/" +  args.relation + "/" + args.relation + ".tsv", 'w')
		ids_2 =  open(args.basedir + "splits/" + args.relation + "/" + args.relation + "_id.tsv", 'w')

		command = "getpairsbyrel" + " " + args.relation + " " + str(1.0)
		pairs_by_rel = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()
		# pairs_by_rel = pairs_by_rel.split('\n')[1:-1][:4000]
		pairs_by_rel = pairs_by_rel.split('\n')[1:-1]
		pairs_by_rel = [pair.split('\t') for pair in pairs_by_rel]
		pairs_by_rel = [[pair[0], pair[1], pair[2], pair[3], pair[4], min(int(pair[5]), int(pair[6]))] for pair in pairs_by_rel]
		
		import operator
		pairs_by_rel = sorted(pairs_by_rel, key=operator.itemgetter(5), reverse = True)		

		for pair in pairs_by_rel:
			ids.write(pair[3] + '\t' + pair[4] + '\t' + args.relation + '\t' + str(pair[5]) +'\n')
			ids_2.write(pair[0] + '\t' + pair[1] + '\t' + pair[2] + '\t' + str(pair[5])  +'\n')
		ids.close()
		ids_2.close()

	with closing(open(args.basedir + "splits/" + args.relation + "/" + args.relation + ".tsv")) as f:
		nodes = f.readlines()
		raw_entities = [nodes[i].rstrip('\n').split('\t')[:3] for i in range(len(nodes))]
		entities = raw_entities[:5000]

	with closing(open(args.basedir + "splits/" + args.relation + "/" + args.relation + "_id.tsv")) as f:
		nodes = f.readlines()
		raw_entities_id = [map(int, nodes[i].rstrip('\n').split('\t')[:2]) for i in range(len(nodes))]
		entities_id = raw_entities_id[:5000]
		unique_id = sorted(set(sum(entities_id, [])))

	if not os.path.isfile(args.basedir + "splits/" + args.relation + "/" + args.relation + "_embedding.npy"):
		file_names = open(args.basedir + "graphs/TransE_entity_vec.txt" , 'r')
		file_name = file_names.readline()
		count = 1
		index = 0

		flag = True
		map_embed = {}
		while (file_name != "") and flag:
			file_name = file_names.readline()
			if count == unique_id[index]:
				em = map(float, file_name.rstrip('\n').split(' ')[:100])
				map_embed[count] = em
				index = index + 1
				if (index == len(unique_id)):
					flag = False
			count = count + 1
		file_names.close()
		entities_embed = [sum([map_embed[i], map_embed[j]], []) for i, j in entities_id]
		entities_embed = np.array(entities_embed)
		np.save(args.basedir + "splits/" + args.relation + "/" + args.relation + "_embedding.npy", entities_embed)

	else:
		entities_embed = np.load(args.basedir + "splits/" + args.relation + "/" + args.relation + "_embedding.npy")

	
	if not os.path.isfile(args.basedir + "splits/" + args.relation + "/" + args.relation + "_clustering.npy"):
		# score = distance.pdist(entities_embed, lambda u, v: np.mean([distance.euclidean(u[:100],v[:100]), 
		#               distance.euclidean(u[:100],v[100:]),
		#               distance.euclidean(u[100:],v[:100]),
		#               distance.euclidean(u[100:],v[100:])]) )
		# score = sklearn.metrics.pairwise.pairwise_distances(entities_embed, metric= semantic_distance, n_jobs=6)
		score = sklearn.metrics.pairwise.pairwise_distances(entities_embed[:,:100], entities_embed[:,:100], n_jobs=6) +  \
				sklearn.metrics.pairwise.pairwise_distances(entities_embed[:,100:], entities_embed[:,100:], n_jobs=6) + \
				sklearn.metrics.pairwise.pairwise_distances(entities_embed[:,100:], entities_embed[:,:100], n_jobs=6) + \
				sklearn.metrics.pairwise.pairwise_distances(entities_embed[:,:100], entities_embed[:,100:], n_jobs=6) 
		score = score/4
		upper_i = np.triu_indices(len(entities_embed),1)
		score = score[upper_i]
		np.save(args.basedir + "splits/" + args.relation + "/" + args.relation + "_clustering.npy", score)
	else:
		score = np.load(args.basedir + "splits/" + args.relation + "/" + args.relation + "_clustering.npy")

	Z = linkage(score,  method="average")


	heterogenity = []
	l_cut = np.arange(np.ceil(Z[0,2]), np.ceil(Z[-1,2]), 0.1)
	for i_cut in l_cut:
		cut = hier.fcluster(Z, i_cut, criterion="distance")
		unique_cut = np.unique(cut)
		heterogenity.append(len(unique_cut))
	
	acceleration = -np.diff(heterogenity)
	acceleration = acceleration*l_cut[1:]

	threshold = l_cut[acceleration.argmax() + 1]

	cut = hier.fcluster(Z, threshold, criterion="distance")
	
	(values,counts) = np.unique(cut,return_counts=True)
	sort_index = np.argsort(-counts)
	values = values[sort_index]
	counts = counts[sort_index]
	
	homogeneity_maj = 0
	consistency_maj = []

	error = ""
	flag_possibility = True

	if args.level == "easy":
		num_positive = int(np.ceil(args.size_scenario*args.pos_neg/(1+args.pos_neg)))
		# positive_set = np.array(entities)[cut==values[0]][:num_positive]
		maj_len = (int) (args.homo*num_positive)
		mij_len = num_positive - maj_len
		mij_mask = np.ones(len(cut), dtype = bool)

		positive_set = np.array(entities)[cut==values[0]]
		np.random.shuffle(positive_set)
		positive_set = positive_set[:maj_len]
		positive_set = np.concatenate((positive_set,np.array([values[0]]*len(positive_set))[np.newaxis].T),axis=1)
		mij_mask[cut==values[0]] = False
		i_cluster = 1
		while (len(positive_set) < maj_len) and (i_cluster<len(values)):
			extra_set = np.array(entities)[cut==values[i_cluster]]
			np.random.shuffle(extra_set)
			extra_set = extra_set[:(maj_len-len(positive_set))]
			extra_set = np.concatenate((extra_set,np.array([values[i_cluster]]*len(extra_set))[np.newaxis].T),axis=1)
			positive_set = np.concatenate((positive_set, extra_set), axis=0)
			mij_mask[cut==values[i_cluster]] = False
			i_cluster = i_cluster + 1
		homogeneity_maj = i_cluster

		mij_cut = np.array(entities)[mij_mask]
		mij_cut = np.concatenate((mij_cut, cut[mij_mask][np.newaxis].T), axis = 1)
		if (mij_len > len(mij_cut)):
			mij_len = len(mij_cut)
		positive_set = np.concatenate((positive_set, mij_cut[np.random.choice(len(mij_cut), mij_len, replace = False),:]), axis=0)		
		positive_set = positive_set.tolist()

		if (len(positive_set) < num_positive):
			flag_possibility = False
			error = "Cannot create scenario: Not enough possible example...."
		else:
			num_negative = args.size_scenario - num_positive
			negative_rate = int(np.ceil(1/args.pos_neg))
			negative_set = []
			for positive_sample in positive_set:
				negative_sample = [(positive_sample[0], tmp[1], tmp[2]) for tmp in positive_set]
				negative_sample = list(set(negative_sample))
				for tmp in raw_entities:
					if (tuple(tmp) in negative_sample):
						negative_sample.remove(tuple(tmp))
				random.shuffle(negative_sample)
				count_neg = 0

				for negative_ in negative_sample:
					tmp = list(negative_)
					tmp.append('*')
					tmp.append('*')
					if (tmp not in negative_set):				
						negative_set.append(tmp)
						count_neg = count_neg + 1
					if count_neg >= negative_rate:
						break
			random.shuffle(negative_set)
			negative_set = negative_set[:num_negative]


	elif args.level == "hard":
		num_positive = int(args.size_scenario*args.pos_neg/(1+args.pos_neg))
		maj_len = (int) (args.homo*num_positive)
		mij_len = num_positive - maj_len
		mij_mask = np.ones(len(cut), dtype = bool)

		args.similar_relation = args.similar_relation[:4]
		if (len(args.similar_relation) > 0):
			consistency_maj.append(len(args.similar_relation))
			i_cluster = 0
			positive_set = []

			while (len(positive_set) < maj_len) and (i_cluster<len(values)):
				cluster_tmp  = np.array(entities)[cut==values[i_cluster]].tolist()
				random.shuffle(cluster_tmp)
				mij_mask[cut==values[i_cluster]] = False
				for positive_sample in cluster_tmp:
					# if (positive_sample in overlapping_pair ):
					positive_sample.append(i_cluster)
					positive_set.append(positive_sample)

					if (len(positive_set) >= maj_len):
						break
				i_cluster = i_cluster + 1 
		
			positive_set = positive_set[:maj_len]
			homogeneity_maj = i_cluster
			# mij_mask[cut==values[0]] = False
			# i_cluster = 1
			# while (len(positive_set) < maj_len):
			# 	positive_set = np.concatenate((positive_set, np.array(entities)[cut==values[i_cluster]][:(maj_len-len(positive_set))]), axis=0)
			# 	mij_mask[cut==values[i_cluster]] = False
			# 	i_cluster = i_cluster + 1
			mij_cut = np.array(entities)[mij_mask].tolist()
			random.shuffle(mij_cut)
			mij_label = cut[mij_mask].tolist()
			if (mij_len > len(mij_cut)):
				mij_len = len(mij_cut)

			i = 0
			while ((len(positive_set) < num_positive) and (i<len(mij_cut))):
				non_homo_sample = mij_cut[i]
				# if (non_homo_sample in overlapping_pair ):
				non_homo_sample.append(mij_label[i])
				positive_set.append(non_homo_sample)
				i = i + 1	

			if (len(positive_set) < num_positive):
				flag_possibility = False
				error = "Cannot create scenario: Not enough overlapping positive examples..."
			else:
				# positive_set = np.concatenate((positive_set, mij_cut[np.random.choice(len(mij_cut), mij_len, replace = False),:]), axis=0)		
				# positive_set = positive_set.tolist()
				random.shuffle(positive_set)

				negative_set = []
				negative_len = args.size_scenario - num_positive
				maj_len = (int) (args.consis*negative_len)
				mij_len = negative_len - maj_len

				possible_pairs = [[i, j] for i in positive_set for j in args.similar_relation]
				random.shuffle(possible_pairs)
				count = 0
				repeat = 0
				negative_cluster = []
				while (len(negative_set) < maj_len) and (repeat<4):		
					positive_sample = possible_pairs[count][0]
					similar_relation = possible_pairs[count][1]

					command = "neighborwithrel" + " " + positive_sample[0] + " " + similar_relation + " " + "TRUE"
					dst_by_rel = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
					dst_by_rel = dst_by_rel.split('\n')[1:-1]
					if (len(dst_by_rel) > 0 ):
						dst_by_rel = [(positive_sample[0], pair.split('\t')[1], positive_sample[2], similar_relation, positive_sample[3]) for pair in dst_by_rel]
						dst_by_rel = set(dst_by_rel)
			
						for j in dst_by_rel:
							if (list(j)[:3] not in raw_entities) and (list(j) not in negative_set):
								negative_set.append(list(j))	
								negative_cluster.append(j[4])
								break		

					command = "neighborwithrel" + " " + positive_sample[1] + " " + similar_relation + " " + "FALSE"
					src_by_rel = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
					src_by_rel = src_by_rel.split('\n')[1:-1]
					if (len(src_by_rel) > 0):
						src_by_rel = [(pair.split('\t')[1], positive_sample[1], positive_sample[2], similar_relation, positive_sample[3]) for pair in src_by_rel]
						src_by_rel = set(src_by_rel)

						for j in src_by_rel:
							if (list(j)[:3] not in raw_entities) and (list(j) not in negative_set):
								negative_set.append(list(j))
								negative_cluster.append(j[4])
								break

					count = count + 1
					if (count == len(possible_pairs)):
						repeat = repeat + 1
						count = 0
					# if len(negative_set) > negative_len:
					# 	break
				random.shuffle(negative_set)
				negative_set = negative_set[:maj_len]

				positive_set_no_label = [x[:3] for x  in positive_set]
				repeat = 0
				i_cluster = 0
				while ((len(negative_set) < maj_len) and (repeat < 100)):
					extra_positive_set = np.array(entities)[cut==values[i_cluster]].tolist()
					extra_positive_set = [sample for sample in extra_positive_set if sample not in positive_set_no_label]
					for similar_relation in args.similar_relation:
						for positive_sample in extra_positive_set:
							command = "neighborwithrel" + " " + positive_sample[0] + " " + similar_relation + " " + "TRUE"
							dst_by_rel = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
							dst_by_rel = dst_by_rel.split('\n')[1:-1]
							if (len(dst_by_rel) > 0 ):
								dst_by_rel = [(positive_sample[0], pair.split('\t')[1], positive_sample[2], similar_relation, values[i_cluster]) for pair in dst_by_rel]
								dst_by_rel = set(dst_by_rel)
								
								for j in dst_by_rel:
									if (list(j)[:3] not in raw_entities) and (list(j) not in negative_set):
										negative_set.append(list(j))	
										negative_cluster.append(j[4])
										# if (len(negative_set) >= maj_len):
										break		

							if (len(negative_set) >= negative_len):
								break
							command = "neighborwithrel" + " " + positive_sample[1] + " " + similar_relation + " " + "FALSE"
							src_by_rel = os.popen("echo \"" + command +"\" | socat -t 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
							src_by_rel = src_by_rel.split('\n')[1:-1]
							if (len(src_by_rel) > 0):
								src_by_rel = [(pair.split('\t')[1], positive_sample[1], positive_sample[2], similar_relation, values[i_cluster]) for pair in src_by_rel]
								src_by_rel = set(src_by_rel)
				
								for j in src_by_rel:
									if (list(j)[:3] not in raw_entities) and (list(j) not in negative_set):
										negative_set.append(list(j))
										negative_cluster.append(j[4])
										# if (len(negative_set) >= negative_len):
										break
							if (len(negative_set) >= maj_len):
								break
						if (len(negative_set) >= maj_len):
							break
					i_cluster = i_cluster + 1
					if (i_cluster == len(values)):
						i_cluster = 0
						repeat = repeat + 1
				negative_set = negative_set[:maj_len]
				negative_cluster = set(negative_cluster)

				consistency_maj.append(len(negative_cluster))

				if (len(negative_set) < maj_len):
					flag_possibility = False
					error = "Cannot create scenario: Not enough overlapping negative examples.. Decrease Consistency ratio..."
				else:
					while (len(negative_set) < negative_len):	
						tmp = [positive_set[np.random.randint(len(positive_set))][0], positive_set[np.random.randint(len(positive_set))][1], positive_set[0][2]]
						if tmp not in raw_entities:
							tmp.append('*')
							tmp.append('*')
							if (tmp not in negative_set):				
								negative_set.append(tmp)
		else:
			flag_possibility = False
			error = "Cannot create scenario: Overlapping examples with multi relations not exists ..."			

	if not os.path.isfile(args.basedir + "splits/" +args.relation + "/" + args.level + "/" + "scenario.tsv"):
		if not os.path.exists(os.path.dirname(args.basedir + "splits/" +args.relation + "/" + args.level + "/" + "scenario.tsv")):
			try:
				os.makedirs(os.path.dirname(args.basedir + "splits/" +args.relation + "/" + args.level + "/" + "scenario.tsv"))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

	ids =  open(args.basedir + "splits/" +args.relation + "/" + args.level + "/" + "scenario.tsv", 'w')
	if (flag_possibility):
		ids.write("---- Summary ---- \n")
		ids.write("Homogeneity:" + "\t" + str(homogeneity_maj) + " clusters \n")
		if (args.level == "hard"):
			ids.write("Consistency: " + "\t" + str(consistency_maj[1]) + " clusters \t from " + str(consistency_maj[0]) + " similar_relations \n")  
		else:
			ids.write("Consistency ignored... \n")	
		ids.write("----------------- \n")
		ids.write("Subject" + '\t' + "Object" + '\t' + "Relation" + '\t' + "Label" + '\t' + "Origin Relation" + "\t" + "Cluster" + '\n')
		for node in positive_set:
			ids.write(node[0] + '\t' + node[1] + '\t' + node[2] + '\t' + '1' + '\t' + "No" + "\t" + str(node[3]) + '\n')
		for node in negative_set:
			ids.write(node[0] + '\t' + node[1] + '\t' + node[2] + '\t' + '-1' + '\t'+ node[3] + '\t'+ str(node[4]) +'\n')
		ids.close()
		print "Successfully generated !!!"
	else:
		# ids.write("---- Summary ---- \n")
		# ids.write(error + "\n")
		# ids.write("----------------- \n")
		# ids.close()
		print error

	if (flag_possibility):	
		scenario = open(args.basedir + "splits/" +args.relation + "/" + args.level + "/" + "scenario.tsv", 'r')
		scenario = scenario.readlines()
		scenario = scenario[5:]

		kf = KFold(n_splits=10, shuffle = True, random_state=233)
		i_fold = 1;
		for i_train, i_test in kf.split(scenario):
			training_name = args.basedir + "splits/" +args.relation + "/" + args.level + '/' + str(i_fold) + '/training.tsv'
			testing_name = args.basedir + "splits/" +args.relation + "/" + args.level + '/' + str(i_fold) + '/testing.tsv'
			train_set = [scenario[ii] for ii in i_train]
			test_set = [scenario[ii] for ii in i_test]
			if not os.path.exists(os.path.dirname(training_name)):
			    os.makedirs(os.path.dirname(training_name))
			if not os.path.exists(os.path.dirname(testing_name)):
			    os.makedirs(os.path.dirname(testing_name))

			training_file = open(training_name, 'w')
			testing_file = open(testing_name, 'w')

			for node in train_set:
			    training_file.write(node)

			for node in test_set:
			    testing_file.write(node)

			training_file.close()
			testing_file.close()

			i_fold+=1



# for i in np.where(cut==values[0])[0][:10]:
#     print("---------------------------------")
#     print(np.array(author_list)[i])
#     subject_em = embed_author[i,:100]
#     object_em = embed_author[i,100:]
    
#     subj_dis = np.sqrt(np.sum((embed_director[:,:100] - subject_em)**2, axis = 1))
#     obj_dis = np.sqrt(np.sum((embed_director[:,100:] - object_em)**2, axis = 1))
    
#     subj_ind = np.argsort(subj_dis)[:3]
#     obj_ind = np.argsort(obj_dis)[:3]
    
#     print(np.array(director_list)[subj_ind])
#     print(np.array(director_list)[obj_ind])
    
