#   Copyright 2016 The Trustees of Indiana University.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Question-answering via confusion matrix """

import sys
from contextlib import closing
import numpy as np
import pandas as pd
import scipy.sparse as sp
from time import time
from datetime import datetime
import traceback
from multiprocessing import Pool, cpu_count

from ..utils import make_weighted, WEIGHT_FUNCTIONS
from ..algorithms.closure import epclosure, epclosuress

from scipy.sparse import SparseEfficiencyWarning
from warnings import filterwarnings

filterwarnings('ignore', category=SparseEfficiencyWarning)

now = datetime.now
WORKER_DATA = {}
max_tasks_per_worker = 500


def populate_parser(parser):
    parser.add_argument('nodespath', metavar='uris', help='node uris')
    parser.add_argument('edgespath', metavar='edges', help='edges(relations) input file')
    parser.add_argument('adjpath', metavar='graph', help='adjacency matrix')
    parser.add_argument('dataset', metavar='testing', help='tesing input file')
    parser.add_argument('-n', '--nprocs', type=int, help='number of processes')
    parser.add_argument('-u', '--undirected', action='store_true',
                        help='use the undirected network')
    parser.add_argument('-k', '--kind', default='ultrametric',
                        choices=['ultrametric', 'metric'],
                        help='the kind of proximity metric')
    parser.add_argument('-w', '--weight', choices=WEIGHT_FUNCTIONS,
                        default='degree',
                        help='Weight type (default: %(default)s)')


def _init_worker(A, kind):
    global WORKER_DATA
    B = A.tocsc()
    WORKER_DATA['A'] = A
    WORKER_DATA['B'] = B
    WORKER_DATA['kind'] = kind
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker(st):
    try:
        global WORKER_DATA
        A = WORKER_DATA['A']
        B = WORKER_DATA['B']
        kind = WORKER_DATA['kind']
        #print st
        #st = st.split('\t')
        source = st[0]
        target = st[1]
        #targets = WORKER_DATA['targets']
        # first, compute closure to all targets
        #D, _ = epclosuress(A, source, B=B, kind=kind)
        #D = D[targets]
        # then, check if any element needs to be recomputed without its edge
        # removal
        #idx, = np.where(D == 1.0)  # direct neighbors, by definition
        #for i in idx:
        #    target = targets[i]
        s = A[source, target]
        A[source, target] = 0.
        B[source, target] = 0.
        A.eliminate_zeros()
        B.eliminate_zeros()
        d, _ = epclosure(A, source, target, B=B, kind=kind)
        A[source, target] = s
        B[source, target] = s
        return d
    except Exception:
        val, ty, tb = sys.exc_info()
        traceback.print_tb(tb)
        raise


def confmatrix(A, sources_targets, nprocs=None, kind='ultrametric'):
    """
    Confusion matrix with edge removal

    Parameters
    ----------
    A : array_like
        Adjacency matrix.

    sources : array_like
        Source node IDs

    targets : array_like or dict or list of lists
        Target node IDs

    nprocs : int
        The number of processes to use. Default: 90% of the available CPUs or
        2, whichever the largest.

    kind : string
        The metric type. See `knowledge_linker.closure.epclosuress`.

    Returns
    -------

    simil : array_like
        Array of similarity values

    """
    if nprocs is None:
        # by default use 90% of available processors or 2, whichever the
        # largest.
        nprocs = max(int(0.9 * cpu_count()), 2)
    # allocate array to be passed as shared memory
    print >> sys.stderr, \
        '{}: copying graph data to shared memory.'.format(now())
    A = sp.csr_matrix(A)

    initargs = (A, kind)
    print >> sys.stderr, \
        '{}: launching pool of {} workers.'.format(now(), nprocs)
    pool = Pool(processes=nprocs, initializer=_init_worker,
                initargs=initargs, maxtasksperchild=max_tasks_per_worker)
    try:
        with closing(pool):
            result = pool.map_async(_worker, sources_targets)
            while not result.ready():
                result.wait(1)
        pool.join()
        if result.successful():
            print >> sys.stderr, '{}: done'.format(now())
            return result.get()
        else:
            print >> sys.stderr, "{}: There was an error in "\
                "the pool.".format(now())
            sys.exit(2)  # ERROR occurred
    except KeyboardInterrupt:
        print "^C"
        pool.terminate()
        sys.exit(1)  # SIGINT received

def getKey(item):
    return item[1]

def main(ns):
    ## print bookkeeping information
    ns.edgetype = 'undirected' if ns.undirected else 'directed'
    print >> sys.stderr, """
{now}:
    metric:   {ns.kind}
    edges:    {ns.edgetype}
    weight:   {ns.weight}
    nodes:    {ns.nodespath}
    graph:    {ns.adjpath}
    testing:  {ns.dataset}
    edges:  {ns.edgespath}""".format(now=now(), ns=ns)

    ## load nodes list
    
    print >> sys.stderr, '{}: reading nodes..'.format(now())
    with closing(open(ns.nodespath)) as f:
        nodes = f.readlines()
    '''
    node_dict = [nodes[i].rstrip('\n').split('\t') for i in range(len(nodes))]
    from collections import OrderedDict
    vertexmap = OrderedDict(( (entity, int(i)-1) for i, entity in sorted(node_dict, key=getKey) ))
    del node_dict    
    '''

    import os.path
    if os.path.isfile(ns.adjpath):
        print >> sys.stderr, '{}: reading graph..'.format(now())
    else:
        print >> sys.stderr, '{}: creating graph..'.format(now())
        with closing(open(ns.edgespath)) as f:
            from itertools import groupby
            from operator import itemgetter
            from ..utils import coo_dtype
            from tqdm import tqdm
            labeled_edge = f.readlines()
            edge_list = [labeled_edge[i].rstrip('\n').split('\t') for i in range(len(labeled_edge))]
            data = []
            for i_ in tqdm(range(len(edge_list))):
                out_entity, in_entity, rel_ = edge_list[i_]
                data.append((int(out_entity)-1, int(in_entity)-1, 1.0))

        del edge_list, labeled_edge
        data = np.asarray(data, dtype=coo_dtype)
        np.save(ns.adjpath, np.asarray(data, dtype=coo_dtype))
        del data
        print >> sys.stderr, 'info: adj written to {}'.format(ns.adjpath)

    A = make_weighted(ns.adjpath, len(nodes), undirected=ns.undirected, weight=ns.weight)

    with closing(open('../data/splits/'+ ns.dataset + '/relations_to_run.tsv')) as f:
        relation = f.readline()

    for i_fold in range(1,11):
        print 'KL Fold ' + str(i_fold) + ' ...'
        testing_name = '../data/splits/'+ ns.dataset + '/' + relation + '/' + str(i_fold) + '/testing_id.tsv'
        with closing(open(testing_name)) as f:
            st = f.readlines()
            testing_set = [st[i].rstrip('\n').split('\t') for i in range(len(st))]
        
        testing_set = [[int(i), int(j), int(l)] for i, j, k, l in testing_set]   
        testing_set_2 = [[i-1, j-1] for i, j, k in testing_set]
        A_copy = A.copy()
        for source, target in testing_set_2:
            A_copy[source, target] = 0.
            A_copy[target, source] = 0.
            A_copy.eliminate_zeros()

        t1 = time()
        B = confmatrix(A_copy, testing_set_2, nprocs=ns.nprocs, kind=ns.kind)
        print 'Time taken: {:.2f}s\n'.format(time() - t1)
        outf = pd.concat([pd.DataFrame(testing_set), pd.DataFrame(B)], axis=1)
        outdirs = '../results/' + ns.dataset + '/KL/' + ns.kind + '/' + relation + '_fold_' + str(i_fold) + '/score.tsv'
        if not os.path.exists(os.path.dirname(outdirs)):
            os.makedirs(os.path.dirname(outdirs))
        outf.to_csv(outdirs, sep='\t', header=False, index=False, encoding='utf-8')
        print '* Saved score results'  
