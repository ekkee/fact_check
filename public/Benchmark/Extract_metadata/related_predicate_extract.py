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

def get_overlapping_pair(target_relation, similar_relation ):
	overlap_pair = []
	if (similar_relation.find("(-1)") >= 0 ):
		command = "getoverlappingpairs" + " " + target_relation + " " + similar_relation.replace("(-1)","")  + " FALSE"
	else:
		command = "getoverlappingpairs" + " " + target_relation + " " + similar_relation  + " TRUE"
	pairs_ = os.popen("echo \"" + command +"\" | socat -t 36000 -T 3600 - UNIX-CONNECT:/tmp/gbserver ").read()			
	pairs_ = pairs_.split('\n')[1:-1]

	return len(pairs_)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
 	parser.add_argument('basedir', metavar='base', help='experiment base directory')
	parser.add_argument('-r', type=str, required=True,
            dest='relation', help='Relation')

	args = parser.parse_args()


	with closing(open(args.basedir + "graphs/edge_dict.tsv")) as f:
		nodes = f.readlines()
		edge_types = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]

	edge_types = [[int(i)-1, j] for i, j in edge_types]
	edge_types = dict(edge_types)

	with closing(open(args.basedir + "graphs/TransE_relation_vec.txt")) as f:
		nodes = f.readlines()
		nodes = nodes[1:]
		embedding = [map(float, x.rstrip('\n').split(' ')[:100]) for x in nodes]

	embedding = np.array(embedding)
	target_index = edge_types.keys()[edge_types.values().index(args.relation)]

	with closing(open(args.basedir + "graphs/ontology_left_predicate.tsv")) as f:
	    dis = f.readlines()
	    left_ontology = [x.rstrip('\n').split('\t') for x in dis]
	    left_ontology = [[x, y.split(',')] for x, y  in left_ontology]
	    left_ontology = dict(left_ontology)
	    
	with closing(open(args.basedir + "graphs/ontology_right_predicate.tsv")) as f:
	    dis = f.readlines()
	    right_ontology = [x.rstrip('\n').split('\t') for x in dis]
	    right_ontology = [[x, y.split(',')] for x, y  in right_ontology]
	    right_ontology = dict(right_ontology)

	right_target = set(right_ontology[args.relation])
	left_target = set(left_ontology[args.relation])

	score = {}
	for i in edge_types.keys():
		relation_rel = edge_types[i]
		if (relation_rel !=  args.relation) and (relation_rel in left_ontology.keys()) and (relation_rel in right_ontology.keys()):
			right_rel = set(right_ontology[relation_rel])
			left_rel = set(left_ontology[relation_rel])			
			right_card = len(right_target.intersection(right_rel))
			left_card = len(left_target.intersection(left_rel))

			score_forward = 0.5*right_card/(len(right_target)+len(right_rel)-2*right_card+1) + \
									0.5*left_card*1.0/(len(left_target)+len(left_rel)-2*left_card+1)
			score_forward = score_forward/distance.euclidean(embedding[target_index], embedding[i])
		
			score[relation_rel] = score_forward
			# right_card = len(right_target.intersection(left_rel))
			# left_card = len(left_target.intersection(right_rel))

			# score_backward = 0.5*right_card/(len(right_target)+len(left_rel)-2*right_card+1) + \
			# 						0.5*left_card*1.0/(len(left_target)+len(right_rel)-2*left_card+1)

			# if (score_forward >= score_backward ):
			# 	score[relation_rel] = score_forward
			# else:
			# 	score["(-1)" + relation_rel] = score_backward
				
			# score[relation_rel] = 0.5*right_card/(len(right_target)+len(right_rel)-2*right_card+1) + \
									# 0.5*left_card*1.0/(len(left_target)+len(left_rel)-2*left_card+1)
			# score[relation_rel] = score[relation_rel]/distance.euclidean(embedding[target_index], embedding[i])							
			#score[relation_rel] = 1/distance.euclidean(embedding[target_index], embedding[i])							
	#score = distance.pdist(embedding)
	import operator
	sorted_score = sorted(score.items(), key=operator.itemgetter(1), reverse = True)
	sorted_score = sorted_score[:4]
	sorted_score = [list(x) for x in sorted_score]

	# for score in sorted_score:
	# 	score.append(get_overlapping_pair(args.relation, score[0]))

	# sorted_score = sorted(sorted_score, key=operator.itemgetter(2), reverse = True)
	# sorted_score = sorted_score[:10]
	




	# Z = linkage(score,  method="average")

	# heterogenity = []
	# l_cut = np.arange(np.ceil(Z[0,2]), np.ceil(Z[-1,2]), 0.1)
	# for i_cut in l_cut:
	# 	cut = hier.fcluster(Z, i_cut, criterion="distance")
	# 	unique_cut = np.unique(cut)
	# 	heterogenity.append(len(unique_cut))
	
	# acceleration = -np.diff(heterogenity)
	# acceleration = acceleration*l_cut[1:]

	# threshold = l_cut[acceleration.argmax() + 1]

	# cut = hier.fcluster(Z, threshold, criterion="distance")
	
	# related_index = np.where(cut==cut[target_index])[0]
	# related_index = related_index[related_index!=target_index]

	# print [[name + ' -- ' + str(score) + ' -- ' + str(count)] for name, score, count in sorted_score]
	print [[name + ' -- ' + str(score) ] for name, score in sorted_score]
	#print sorted_score


