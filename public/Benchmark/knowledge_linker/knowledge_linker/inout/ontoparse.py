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

''' Extracts subclass relations from the DBPedia ontology and print them to
console in NT format '''

from lxml import etree
from argparse import ArgumentParser

ns = {
        'owl' :
            'http://www.w3.org/2002/07/owl#',
        'rdf' :
            'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs':
            'http://www.w3.org/2000/01/rdf-schema#'
}


def main():
    parser = ArgumentParser()
    parser.add_argument('owlfile')
    args = parser.parse_args()

    tree = etree.parse(args.owlfile)
    root = tree.getroot()

    # for each owl:Class element
    for onto_elem in root.xpath('owl:Class', namespaces=ns):

        # get the element in the DBPedia ontology
        onto_elem_about, = onto_elem.xpath('@rdf:about', namespaces=ns)

        # skip the first node (owl:Ontology)
        if onto_elem_about == '':
            continue

        # get the children with an rdf:resource attribute
        for child in onto_elem.getchildren():

            child_rdf_resource = child.xpath('@rdf:resource', namespaces=ns)
            if child_rdf_resource:
                # lxml delimits the namespace part of the tag with curly braces,
                # which we don't want in the final output
                child_tag_uri = child.tag.translate(None, '{}')

                nt_item = (onto_elem_about, child_tag_uri, child_rdf_resource[0])
                print '<%s> <%s> <%s> .' % nt_item


if __name__ == '__main__':
    main()
