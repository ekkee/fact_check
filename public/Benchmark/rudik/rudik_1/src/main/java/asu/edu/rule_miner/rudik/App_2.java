package asu.edu.rule_miner.rudik;

import java.time.Instant;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.io.FileNotFoundException;
import java.util.Scanner; 
import java.util.Iterator;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;

import java.util.HashMap;
 import java.util.Map.Entry;
 import java.util.stream.Collectors;
import java.util.LinkedHashMap;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Maps;
import org.apache.jena.ext.com.google.common.collect.Sets;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.jena.ext.com.google.common.collect.Lists;

import asu.edu.rule_miner.rudik.configuration.ConfigurationFacility;
import asu.edu.rule_miner.rudik.model.horn_rule.HornRule;
import asu.edu.rule_miner.rudik.model.statistic.StatisticsContainer;
import asu.edu.rule_miner.rudik.rule_generator.DynamicPruningRuleDiscovery;

//import org.apache.logging.log4j.LogManager;
//import org.apache.logging.log4j.Logger;
import asu.edu.rule_miner.rudik.model.horn_rule.HornRule;
import asu.edu.rule_miner.rudik.model.horn_rule.MultipleGraphHornRule;
import asu.edu.rule_miner.rudik.model.horn_rule.RuleAtom;
import asu.edu.rule_miner.rudik.rule_generator.score.DynamicScoreComputation;

import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.ResultSet;
import org.apache.jena.query.QuerySolution;


// xml library
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.OutputKeys;
import org.xml.sax.SAXException;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class App_2
{

	public static String toDbpedia_resource(String resource) {
		final String result = "http://dbpedia.org/resource/" + resource;
	    return result;
	} 
	public static String toDbpedia_property(String property) {
	    return "http://dbpedia.org/property/" + property;
	} 
    
    public static final List<Set<Pair<Set<RuleAtom>,Double>>> exe_posrule_conf = new ArrayList<Set<Pair<Set<RuleAtom>,Double>>>();
    // public static final List<Pair<Set<RuleAtom>,Double>> all_posrules = new ArrayList<Pair<Set<RuleAtom>,Double>>();
    public static final Set<Pair<Set<RuleAtom>,Double>> current_posrules = Sets.newHashSet();
    public static final Set<Pair<Set<RuleAtom>,Double>> new_posrules = Sets.newHashSet();
    public static final Set<Pair<Set<RuleAtom>,Double>> extend_posrules = Sets.newHashSet();
    public static final Set<Pair<Set<RuleAtom>,Double>> current_negrules = Sets.newHashSet();
    public static final Set<Pair<Set<RuleAtom>,Double>> new_negrules = Sets.newHashSet();
    public static final Set<Pair<Set<RuleAtom>,Double>> extend_negrules = Sets.newHashSet();
    public static final List<List<String>> exe_subjectTypes = new ArrayList<List<String>>();
    public static final List<List<String>> exe_objectTypes = new ArrayList<List<String>>();
	public static final String graph = " <http://dbpedia.org> ";
	public static final String type_prefix = " <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ";
	public static final String ontology_prefix = "http://dbpedia.org/ontology";
	public static final String agent_type = "http://dbpedia.org/ontology/Agent";


 	public static final Set<String> relations = Sets.newHashSet("http://dbpedia.org/ontology/spouse");
	public static final String typeSubject = "http://dbpedia.org/ontology/Person";
	public static final String typeObject = "http://dbpedia.org/ontology/Person";


	public static boolean double_check(String subject, String object, String relation, boolean label) {
		final String s_query = "SELECT ?o  FROM <http://dbpedia.org>" + " WHERE {" + " <" +  subject + "> " + " <" +  relation + "> " + " ?o }  ";
	    QueryExecution qexec = QueryExecutionFactory.sparqlService("http://localhost:8890/sparql", s_query); // http://localhost:8890/sparql
	    final ResultSet s_results = qexec.execSelect();  
      	Set<String> s_set = Sets.newHashSet();
       
        while (s_results.hasNext()) {            
              QuerySolution qs = s_results.next();
              Iterator<String> itVars = qs.varNames();
              while (itVars.hasNext()) {
                  String szVar = itVars.next().toString();
                  String szVal = qs.get(szVar).toString();
                  s_set.add(szVal);                
              } 
          
      	}
      	qexec.close();

		final String o_query = "SELECT ?o  FROM <http://dbpedia.org>" + " WHERE { ?o " + " <" +  relation + "> " + " <" +  object + "> " + " }  ";
	    qexec = QueryExecutionFactory.sparqlService("http://localhost:8890/sparql", o_query); // http://localhost:8890/sparql
	    final ResultSet o_results = qexec.execSelect(); 
      	Set<String> o_set = Sets.newHashSet();
        
        while (o_results.hasNext()) {            
              QuerySolution qs = o_results.next();
              Iterator<String> itVars = qs.varNames();
              while (itVars.hasNext()) {
                  String szVar = itVars.next().toString();
                  String szVal = qs.get(szVar).toString();
                  o_set.add(szVal);                
              } 
          
      	}
      	qexec.close();

      	if (label) {
      		if (s_set.contains(object) || o_set.contains(subject))
      			return true;
      		else
      			return false;
      	}
      	else {
      		if ((s_set.size()>0 || o_set.size()>0) && ((!s_set.contains(object)) && (!o_set.contains(subject))))
      			return true;
      		else
      			return false;
      	}

	}

	public static Set<String> get_type(String sparqlQuery) {

	    QueryExecution qexec = QueryExecutionFactory.sparqlService("http://localhost:8890/sparql", sparqlQuery); // http://localhost:8890/sparql

	    //qexec.setTimeout(20000);
	    ResultSet results = qexec.execSelect();
	    final Set<String> setTypes = Sets.newHashSet();
	    
	    if (!results.hasNext()) {
	      setTypes.add("empty_result");
	    }  
	    else {    
	        while (results.hasNext()) {
	         
	            
	            // Get Result
	            final QuerySolution qs = results.next();
	            
	            // Get Variable Names
	            final Iterator<String> itVars = qs.varNames();

	            // Display Result
	            while (itVars.hasNext()) {
	                final String szVar = itVars.next().toString();
	                final String szVal = qs.get(szVar).toString();
	                // if (szVal.contains(ontology_prefix)) 
	                    setTypes.add(szVal);
	                //System.out.println("[" + szVar + "]: " + szVal);
	            } 
	        }
	    }
	    qexec.close();
	    return setTypes;
	}


	public static int get_pair(String sparqlQuery) {

	    QueryExecution qexec = QueryExecutionFactory.sparqlService("http://localhost:8890/sparql", sparqlQuery); // http://localhost:8890/sparql

	    //qexec.setTimeout(20000);
	    ResultSet results = qexec.execSelect();
	    final Set<Pair<String,String>> setTypes = Sets.newHashSet();
	    
	    if (!results.hasNext()) {
	      qexec.close();
	      return 0;
	    }  
	    else {    
	    	qexec.close();
	    	return 1;
	    	// int count = 0;
		    // while(results.hasNext()){
		    //   count = count + 1;
		    //   final QuerySolution oneResult = results.next();
		    //   final String subject = oneResult.get("subject").toString();
		    //   final String object = oneResult.get("object").toString();

		      
		    //   if(subject!=null && object!=null && (examples==null || examples.contains(Pair.of(subject, object)))){
		    //   // if(subject!=null && object!=null) {
		    //     setTypes.add(Pair.of(subject, object));
		    //   }
		    //   else if (subject!=null && object!=null && (examples==null || examples.contains(Pair.of(object, subject)))){
		    //   // if(subject!=null && object!=null) {
		    //     setTypes.add(Pair.of(object, subject));
		    //   } 

		    // }
		    // System.out.println("Count " + count);
	    }
	    
	    // return setTypes;
	}

	public static void write_xml(String relation, String xmlFilePath, Map<String, Integer> subjectTypes, Map<String, Integer> objectTypes, Map<HornRule, Double> outputRules_positive, Map<HornRule, Double> outputRules_negative, int threshold) {

		try {	
			List<HornRule> rule_positive = (outputRules_positive != null) ? Lists.newArrayList(outputRules_positive.keySet()) : null;
			List<HornRule> rule_negative = (outputRules_negative!= null) ? Lists.newArrayList(outputRules_negative.keySet()) : null;

			DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();

			DocumentBuilder documentBuilder = documentFactory.newDocumentBuilder();

			Document document = null;

			File xml = new File(xmlFilePath);
			if (xml.exists() && xml.isFile()) 
				document = documentBuilder.parse(xml);			
			else
				document = documentBuilder.newDocument();
			Element root = null;
			if (document.getElementsByTagName(relation).getLength() >0 ) {
				NodeList restElmLst = document.getElementsByTagName(relation);
    			root = (Element)restElmLst.item(0);   			
			}
			else {
				root = document.createElement(relation);
				document.appendChild(root);

			}
			
			
			Element sub_root = document.createElement("Execution");
			root.appendChild(sub_root);

			Element stypes = document.createElement("Subject_Types");

			sub_root.appendChild(stypes);

			List<String> list_subjectTypes = (subjectTypes!= null) ? Lists.newArrayList(subjectTypes.keySet()) : null;
			for(final String type:list_subjectTypes.subList(0,4)){
					Element t = document.createElement("type");
					t.appendChild(document.createTextNode(type));
					stypes.appendChild(t);			
			}

			Element otypes = document.createElement("Object_Types");

			sub_root.appendChild(otypes);

			List<String> list_objectTypes = (objectTypes!= null) ? Lists.newArrayList(objectTypes.keySet()) : null;
			for(final String type:list_objectTypes.subList(0,4)){
					Element t = document.createElement("type");
					t.appendChild(document.createTextNode(type));
					otypes.appendChild(t);			
			}

			Element p_rules = document.createElement("Positive_Rules");
			sub_root.appendChild(p_rules);
			for(final HornRule rule:rule_positive){
				if (outputRules_positive.get(rule) != null) {
					Element t = document.createElement("rule");
					p_rules.appendChild(t);		
					Element t1 = document.createElement("atoms");
					t1.appendChild(document.createTextNode(rule.toString()));	
					t.appendChild(t1);
					Element t2 = document.createElement("confidence");
					t2.appendChild(document.createTextNode(Double.toString(outputRules_positive.get(rule))));	
					t.appendChild(t2);						
				}
			}

			Element n_rules = document.createElement("Negative_Rules");
			sub_root.appendChild(n_rules);
			for(final HornRule rule:rule_negative){
				if (outputRules_negative.get(rule) != null) {
					Element t = document.createElement("rule");
					n_rules.appendChild(t);		
					Element t1 = document.createElement("atoms");
					t1.appendChild(document.createTextNode(rule.toString()));	
					t.appendChild(t1);
					Element t2 = document.createElement("confidence");
					t2.appendChild(document.createTextNode(Double.toString(outputRules_negative.get(rule))));	
					t.appendChild(t2);						
				}
			}

			// create the xml file
			//transform the DOM Object to an XML File
			TransformerFactory transformerFactory = TransformerFactory.newInstance();
			Transformer transformer = transformerFactory.newTransformer();
			transformer.setOutputProperty(OutputKeys.VERSION, "1.0");
			DOMSource domSource = new DOMSource(document);
			StreamResult streamResult = new StreamResult(new File(xmlFilePath));

			// If you use
			// StreamResult result = new StreamResult(System.out);
			// the output will be pushed to the standard output ...
			// You can use that for debugging 

			transformer.transform(domSource, streamResult);	

		} catch (ParserConfigurationException pce) {
			pce.printStackTrace();
		} catch (TransformerException tfe) {
			tfe.printStackTrace();
		} catch (SAXException sfe) {
			sfe.printStackTrace();
		} catch (IOException ife) {
			ife.printStackTrace();
		}
	
	}


	public static void write_extended_rules(String relation, String xmlFilePath,  Set<Pair<Set<RuleAtom>,Double>> ex_posRules, Set<Pair<Set<RuleAtom>,Double>> ex_negRules) {

		try {	


			DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();

			DocumentBuilder documentBuilder = documentFactory.newDocumentBuilder();

			Document document = null;

			File xml = new File(xmlFilePath);
			if (xml.exists() && xml.isFile()) 
				document = documentBuilder.parse(xml);			
			else
				document = documentBuilder.newDocument();
			Element root = null;
			if (document.getElementsByTagName(relation).getLength() >0 ) {
				NodeList restElmLst = document.getElementsByTagName(relation);
    			root = (Element)restElmLst.item(0);   			
			}
			else {
				root = document.createElement(relation);
				document.appendChild(root);

			}
			
			
			Element sub_root = document.createElement("Execution");
			root.appendChild(sub_root);

			

			Element p_rules = document.createElement("New_Positive_Rules");
			sub_root.appendChild(p_rules);

		    Iterator<Pair<Set<RuleAtom>,Double>> iter = ex_posRules.iterator();
		    while (iter.hasNext()) {
		    	Pair<Set<RuleAtom>,Double> rule_score = iter.next();
		    	Set<RuleAtom> rule = rule_score.getLeft();
		    	double conf_val = rule_score.getRight();
				
				String hornRule = "";
				Iterator<RuleAtom> sub_iter = rule.iterator();
				while (sub_iter.hasNext())
					hornRule+=sub_iter.next()+" & ";			
				hornRule = hornRule.substring(0,hornRule.length()-3);

				Element t = document.createElement("rule");
				p_rules.appendChild(t);		
				Element t1 = document.createElement("atoms");
				t1.appendChild(document.createTextNode(hornRule));	
				t.appendChild(t1);
				Element t2 = document.createElement("confidence");
				t2.appendChild(document.createTextNode(Double.toString(conf_val)));	
				t.appendChild(t2);	

		    }



			Element n_rules = document.createElement("New_Negative_Rules");
			sub_root.appendChild(n_rules);

		    iter = ex_negRules.iterator();
		    while (iter.hasNext()) {
		    	Pair<Set<RuleAtom>,Double> rule_score = iter.next();
		    	Set<RuleAtom> rule = rule_score.getLeft();
		    	double conf_val = rule_score.getRight();

				String hornRule = "";
				Iterator<RuleAtom> sub_iter = rule.iterator();
				while (sub_iter.hasNext())
					hornRule+=sub_iter.next()+" & ";	
				hornRule = hornRule.substring(0,hornRule.length()-3);

				Element t = document.createElement("rule");
				n_rules.appendChild(t);		
				Element t1 = document.createElement("atoms");
				t1.appendChild(document.createTextNode(hornRule));	
				t.appendChild(t1);
				Element t2 = document.createElement("confidence");
				t2.appendChild(document.createTextNode(Double.toString(conf_val)));	
				t.appendChild(t2);	
		    }



			// create the xml file
			//transform the DOM Object to an XML File
			TransformerFactory transformerFactory = TransformerFactory.newInstance();
			Transformer transformer = transformerFactory.newTransformer();
			transformer.setOutputProperty(OutputKeys.VERSION, "1.0");
			DOMSource domSource = new DOMSource(document);
			StreamResult streamResult = new StreamResult(new File(xmlFilePath));

			// If you use
			// StreamResult result = new StreamResult(System.out);
			// the output will be pushed to the standard output ...
			// You can use that for debugging 

			transformer.transform(domSource, streamResult);	

		} catch (ParserConfigurationException pce) {
			pce.printStackTrace();
		} catch (TransformerException tfe) {
			tfe.printStackTrace();
		} catch (SAXException sfe) {
			sfe.printStackTrace();
		} catch (IOException ife) {
			ife.printStackTrace();
		}
	
	}



	public static void load_rules_from_xml(String xmlFilePath, DynamicPruningRuleDiscovery naive) {
	    try {

			File fXmlFile = new File(xmlFilePath);
			DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
			Document doc = dBuilder.parse(fXmlFile);
					
			doc.getDocumentElement().normalize();

			System.out.println("Root element :" + doc.getDocumentElement().getNodeName());
					
			NodeList nList = doc.getElementsByTagName("Execution");
					
			// System.out.println("----------------------------");

			for (int temp = 0; temp < nList.getLength(); temp++) {

				Node nNode = nList.item(temp);
						
				System.out.println("\nCurrent Element :" + nNode.getNodeName());
						
				// if (nNode.getNodeType() == Node.ELEMENT_NODE) {

				Element eElement = (Element) nNode;

				Element sElement = (Element) eElement.getElementsByTagName("Subject_Types").item(0);
				List<String> subjectTypeSet =  new ArrayList<String>();
				NodeList sNodelist = sElement.getElementsByTagName("type");

				for (int i = 0; i < sNodelist.getLength(); i ++) {
					subjectTypeSet.add(sNodelist.item(i).getTextContent());
				}
				exe_subjectTypes.add(subjectTypeSet);

				sElement = (Element) eElement.getElementsByTagName("Object_Types").item(0);
				List<String> objectTypeSet = new ArrayList<String>();
				sNodelist = sElement.getElementsByTagName("type");

				for (int i = 0; i < sNodelist.getLength(); i ++) {
					objectTypeSet.add(sNodelist.item(i).getTextContent());
				}
				exe_objectTypes.add(objectTypeSet);

				sElement = (Element) eElement.getElementsByTagName("Positive_Rules").item(0);
				sNodelist = sElement.getElementsByTagName("rule");
				Set<Pair<Set<RuleAtom>,Double>> rule_conf = Sets.newHashSet();
				for (int i = 0; i < sNodelist.getLength(); i ++) {
					Element sEle = (Element) sNodelist.item(i);
					String e_rule =  sEle.getElementsByTagName("atoms").item(0).getTextContent();
					System.out.println(e_rule);
					double e_conf = Double.parseDouble(sEle.getElementsByTagName("confidence").item(0).getTextContent());
					if (e_conf>0.2) {
						Set<RuleAtom> e_hornRule = HornRule.readHornRule(e_rule);
						rule_conf.add(Pair.of(e_hornRule, e_conf));
						current_posrules.add(Pair.of(e_hornRule, e_conf));
						Set<RuleAtom> new_hornRule = Sets.newHashSet(e_hornRule);
						new_hornRule.add(new RuleAtom("subject", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", subjectTypeSet.get(0)));
						if (e_conf > 0.2) {
							Pair<Integer, Integer> support = naive.getRuleSupport(e_hornRule, relations, typeSubject, typeObject, true);
							Pair<Integer, Integer> new_support = naive.getRuleSupport(new_hornRule, relations, typeSubject, typeObject, true);
							if (5*new_support.getLeft()>support.getLeft()) {
								double new_conf = 1.0*new_support.getLeft()/(new_support.getLeft()+new_support.getRight());
								if (new_conf>e_conf)
									extend_posrules.add(Pair.of(new_hornRule, new_conf));
							}
						}

						// new_posrules.add(Pair.of(new_hornRule, e_conf));
					}
				}
				exe_posrule_conf.add(rule_conf);

				sElement = (Element) eElement.getElementsByTagName("Negative_Rules").item(0);
				sNodelist = sElement.getElementsByTagName("rule");
				rule_conf = Sets.newHashSet();
				for (int i = 0; i < sNodelist.getLength(); i ++) {
					Element sEle = (Element) sNodelist.item(i);
					String e_rule =  sEle.getElementsByTagName("atoms").item(0).getTextContent();
					System.out.println(e_rule);
					double e_conf = Double.parseDouble(sEle.getElementsByTagName("confidence").item(0).getTextContent());
					if ((e_conf>0.2) && (e_conf<0.93)) {
						Set<RuleAtom> e_hornRule = HornRule.readHornRule(e_rule);
						rule_conf.add(Pair.of(e_hornRule, e_conf));
						current_negrules.add(Pair.of(e_hornRule, e_conf));
						Set<RuleAtom> new_hornRule = Sets.newHashSet(e_hornRule);
						new_hornRule.add(new RuleAtom("subject", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "v_type"));
						new_hornRule.add(new RuleAtom("v_type","!=",subjectTypeSet.get(0)));

						Pair<Integer, Integer> support = naive.getRuleSupport(e_hornRule, relations, typeSubject, typeObject, false);
						Pair<Integer, Integer> new_support = naive.getRuleSupport(new_hornRule, relations, typeSubject, typeObject, false);
						if (5*new_support.getLeft()>support.getLeft()) {
							double new_conf = 0.01*new_support.getLeft()/(0.01*new_support.getLeft()+new_support.getRight());
							if (new_conf>e_conf)
								extend_negrules.add(Pair.of(new_hornRule, new_conf));
						}						
						// new_negrules.add(Pair.of(new_hornRule, e_conf));
						Set<RuleAtom> new_hornRule_2 = Sets.newHashSet(e_hornRule);
						new_hornRule_2.add(new RuleAtom("subject", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", subjectTypeSet.get(0)));
						
						new_support = naive.getRuleSupport(new_hornRule_2, relations, typeSubject, typeObject, false);
						if (5*new_support.getLeft()>support.getLeft()) {
							double new_conf = 0.01*new_support.getLeft()/(0.01*new_support.getLeft()+new_support.getRight());
							if (new_conf>e_conf)
								extend_negrules.add(Pair.of(new_hornRule_2, new_conf));
						}	
						// new_negrules.add(Pair.of(new_hornRule_2, e_conf));
					}
				}
			

				// }
			}
			// System.out.println("----------------------------");
			// System.out.println(all_posrules);
	    } catch (Exception e) {
			e.printStackTrace();
	    }
	}
	private final static Logger LOGGER = LoggerFactory.getLogger(App.class.getName());
	public static void main( String[] args ) throws IOException
	{

		String sparqlQuery = "";

	    String path_file = "";
	    try {
	        path_file = args[0];
	    }
	    catch (Exception e) {
	        System.out.println("No arguments");
	        System.exit(1);
	    }
	
    	// final Set<String> relations = Sets.newHashSet("http://dbpedia.org/ontology/nearestCity");
    	// final String typeSubject = "http://dbpedia.org/ontology/Place";
    	// final String typeObject = "http://dbpedia.org/ontology/PopulatedPlace";



     	// final Set<String> relations = Sets.newHashSet("http://dbpedia.org/ontology/employer");
    	// final String typeSubject = "http://dbpedia.org/ontology/Person";
    	// final String typeObject = "http://dbpedia.org/ontology/Organisation";
        // final Set<String> relations = Sets.newHashSet("http://dbpedia.org/ontology/spouse");
    	// final String typeSubject = "http://dbpedia.org/ontology/Person";
    	// final String typeObject = "http://dbpedia.org/ontology/Person";

        // final String typeSubject = null;
        // final String typeObject = null;
     	// final Set<String> relations = Sets.newHashSet("http://dbpedia.org/ontology/manufacturer");
    	// final String typeSubject = "http://dbpedia.org/ontology/MeanOfTransportation";
    	// final String typeObject = "http://dbpedia.org/ontology/Organisation";

    	DynamicPruningRuleDiscovery naive = new DynamicPruningRuleDiscovery();
    	

	   	final Set<Pair<String,String>> negativeExamples = Sets.newHashSet();
	    final Set<Pair<String,String>> positiveExamples = Sets.newHashSet();


	    // Set<String> subjectSet = Sets.newHashSet();
	    // Set<String> objectSet = Sets.newHashSet();

	    // Map<String, Integer> entity2types_sub = Maps.newHashMap();
	    // Map<String, Integer> entity2types_obj = Maps.newHashMap();

	    Map<String, Integer> entity2types_sub = new HashMap<String, Integer>();
	    Map<String, Integer> entity2types_obj = new HashMap<String, Integer>();

	    File file = new File(path_file); 
	    try {
	        // Scanner sc = new Scanner(file); 
	        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF-8"));
	        // String tmp = sc.nextLine();
	        String tmp = br.readLine();
	        // while (sc.hasNextLine()) {
	        while ((tmp = br.readLine()) != null) {
	            // String[] sample = sc.nextLine().split("\t"); 
	            String[] sample = tmp.split("\t"); 
	            final Pair<String,String> pair = Pair.of(toDbpedia_resource(sample[0]), toDbpedia_resource(sample[1]));

	            ///
                Set<String> currentTypes_sub = naive.get_type(toDbpedia_resource(sample[0]));
                Set<String> currentTypes_obj = naive.get_type(toDbpedia_resource(sample[1]));

                // entity2types.put(subject, currentTypes);
                if (currentTypes_sub.size() == 0 || currentTypes_obj.size() ==0) {
                    System.out.println("Sample " + sample[0] + "-" + sample[1] + " has no types " );

                }
                else {

	                ///

		            if (sample[3].equals("1")) {
		            	if (double_check(pair.getLeft(), pair.getRight(),relations.iterator().next(), true)) {
		                	positiveExamples.add(pair);
		                	for (final String type_ : currentTypes_sub) 
	                			if (type_.contains(ontology_prefix) && !type_.contains(agent_type) && !type_.contains(typeSubject)) 
	                    			entity2types_sub.put(type_, entity2types_sub.getOrDefault(type_, 0) + 1);
			                for (final String type_ : currentTypes_obj) 
		    	            	if (type_.contains(ontology_prefix) && !type_.contains(agent_type) && !type_.contains(typeObject)) 
	            		        	entity2types_obj.put(type_, entity2types_obj.getOrDefault(type_, 0) + 1);
		            	}
		            }
		            else {
		            	if (double_check(pair.getLeft(), pair.getRight(),relations.iterator().next(), false)) {
		                	negativeExamples.add(pair);
		                	for (final String type_ : currentTypes_sub) 
	                			if (type_.contains(ontology_prefix) && !type_.contains(agent_type) && !type_.contains(typeSubject)) 
	                    			entity2types_sub.put(type_, entity2types_sub.getOrDefault(type_, 0) + 1);
			                for (final String type_ : currentTypes_obj) 
		    	            	if (type_.contains(ontology_prefix) && !type_.contains(agent_type) && !type_.contains(typeObject)) 
	            		        	entity2types_obj.put(type_, entity2types_obj.getOrDefault(type_, 0) + 1);
		            	}
		            }

		            // subjectSet.add(toDbpedia_resource(sample[0]));
	            	// objectSet.add(toDbpedia_resource(sample[1]));
            	}
            
	        }
	        System.out.println(" Length Positive Set : " + positiveExamples.size() );
	        System.out.println(" Length Negative Set : " + negativeExamples.size() );  

			// String subjectKey = null;
			// for(String key:entity2types_sub.keySet())
			//   	if(subjectKey==null||entity2types_sub.get(key)>entity2types_sub.get(subjectKey))
			//     	subjectKey=key;

			// String objectKey = null;
			// for(String key:entity2types_obj.keySet())
			//   	if(objectKey==null||entity2types_obj.get(key)>entity2types_obj.get(objectKey))
			//     	objectKey=key;
			entity2types_sub = 
			     entity2types_sub.entrySet().stream()
			    .sorted(Entry.<String, Integer>comparingByValue().reversed())
			    .collect(Collectors.toMap(Entry::getKey, Entry::getValue,
			                              (e1, e2) -> e1, LinkedHashMap::new));
			entity2types_obj = 
			     entity2types_obj.entrySet().stream()
			    .sorted(Entry.<String, Integer>comparingByValue().reversed())
			    .collect(Collectors.toMap(Entry::getKey, Entry::getValue,
			                              (e1, e2) -> e1, LinkedHashMap::new));
	        System.out.println(" Subject Types : " + entity2types_sub);
			System.out.println(" Object Types : " + entity2types_obj);



	  //       System.out.println(" Subject Types : " + subjectKey);
			// System.out.println(" Object Types : " + objectKey);

			// int count_n = 0;
	  //       for (final  Pair<String,String> example  :positiveExamples) {
	  //           try {
	  //               // sparqlQuery = "SELECT ?o  FROM " + graph + " WHERE {?o <http://dbpedia.org/ontology/parent> " + "<"+ example.getLeft() + ">" + ". " + "?o <http://dbpedia.org/ontology/parent> " + "<"+ example.getRight() + ">" + " . }  ";
	  //           	sparqlQuery = "SELECT ?o  FROM " + graph + " WHERE { " + "<"+ example.getLeft() + ">" + " <http://dbpedia.org/ontology/birthPlace> ?o" +  " . " + "<"+ example.getRight() + ">" + " <http://dbpedia.org/ontology/birthPlace> ?o" +  " . }  ";
	  //               Set<String> currentTypes = get_type(sparqlQuery);
	  //               // for (final String type_ : currentTypes) 
	  //               // 	entity2types.put(type_, entity2types.getOrDefault(type_, 0) + 1);
	    	            	
	  //               if (currentTypes.contains("empty_result"))
	  //               	count_n = count_n + 1;
	  //              	else 
	  //              		System.out.println("+ : " + example.getLeft() + " -- " + example.getRight() );
	  //                   // System.out.println("Entity " + subject.toString() + " has no types " );
	  //               // System.out.println("Entity " + subject.toString() + " : " + currentTypes);
	  //           }
	  //           catch (final Exception e) {
	  //               System.out.println("Thread with entity was not able to complet its job." + e);
	  //           }
	  //       }
			/*
			int count_p = 0;
	        for (final  Pair<String,String> example  :positiveExamples) {
	            try {
	                // sparqlQuery = "SELECT ?o  FROM " + graph + " WHERE {?o <http://dbpedia.org/ontology/parent> " + "<"+ example.getLeft() + ">"+ ". " + "?o <http://dbpedia.org/ontology/parent> " + "<"+ example.getRight() + ">" + " . }  ";
					sparqlQuery = "SELECT ?o  FROM " + graph + " WHERE { " + "<"+ example.getLeft() + ">" + " <http://dbpedia.org/ontology/birthPlace> ?o" +  " . " + "<"+ example.getRight() + ">" + " <http://dbpedia.org/ontology/birthPlace> ?o" +  " . }  ";
	                Set<String> currentTypes = get_type(sparqlQuery);
	                // for (final String type_ : currentTypes) 
	                // 	entity2types.put(type_, entity2types.getOrDefault(type_, 0) + 1);
	    	            	
	                if (!currentTypes.contains("empty_result"))
	                	count_p = count_p + 1;
	               	// else
	               		// System.out.println("- : " + example.getLeft() + " -- " + example.getRight() + " -- " + currentTypes.iterator().next() );
	                    // System.out.println("Entity " + subject.toString() + " has no types " );
	                // System.out.println("Entity " + subject.toString() + " : " + currentTypes);
	            }
	            catch (final Exception e) {
	                System.out.println("Thread with entity  was not able to complet its job." + e);
	            }
	        }

	  //       for (final  Pair<String,String> example  : negativeExamples) {
	  //       	System.out.println("- new : " + example.getLeft() + " -- " + example.getRight() );
	  //       }
	  //       System.out.println(" Positive Coverage : " + count_n);
			System.out.println(" Positive Coverage : " + count_p);

			*/
			/*
			sparqlQuery = "SELECT DISTINCT ?subject ?object  FROM <http://dbpedia.org>" 
				+ " WHERE { "
				// + " ?object <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person>.  "
				// + " ?subject <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person>. "

				// + " {{?subject ?targetRelation ?realObject.} UNION  {?realSubject ?targetRelation ?object.}}   "
				// + " ?subject ?otherRelation ?object.   "
				// + " FILTER (?targetRelation = <http://dbpedia.org/ontology/spouse>)   "
				// + " FILTER (?otherRelation = <http://dbpedia.org/ontology/spouse>)   "
				// + " FILTER NOT EXISTS {?subject <http://dbpedia.org/ontology/spouse> ?object.}  "
				+ " ?subject <http://dbpedia.org/ontology/birthPlace> ?v0 . "
				+ " ?object <http://dbpedia.org/ontology/birthPlace> ?v0 . "

				// + " FILTER NOT EXISTS {?subject <http://dbpedia.org/ontology/birthPlace> ?otherv1. "
				// + " ?object <http://dbpedia.org/ontology/birthPlace> ?otherv1. } "
				// + " FILTER (?subject = <http://dbpedia.org/resource/Geoffrey_Cheshire>)"
				// + " FILTER (?object = <http://dbpedia.org/resource/Leonard_Cheshire>)"
				+ "  }  ";


				negativeExamples.add(Pair.of(toDbpedia_resource("Graeme_Fowler"), toDbpedia_resource("Chris_Norbury")));


			final Set<Pair<String,String>> overlapping = get_pair(sparqlQuery, negativeExamples);
			System.out.println("- Size overlapping : " + overlapping.size());

			*/

			/*
			int count_n = 0;
	        for (final  Pair<String,String> example  : negativeExamples) {
	        	
	        	sparqlQuery = "SELECT DISTINCT ?subject ?object  FROM <http://dbpedia.org>" 
					+ " WHERE { "
					+ " ?object <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person>.  "
					+ " ?subject <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person>. "

					+ " {{?subject ?targetRelation ?realObject.} UNION  {?realSubject ?targetRelation ?object.}}   "
					// + " ?subject ?otherRelation ?object.   "
					+ " FILTER (?targetRelation = <http://dbpedia.org/ontology/spouse>)   "
					// + " FILTER (?otherRelation = <http://dbpedia.org/ontology/spouse>)   "
					// + " FILTER NOT EXISTS {?subject <http://dbpedia.org/ontology/spouse> ?object.}  "
					+ " ?subject <http://dbpedia.org/ontology/birthPlace> ?v0 . "
					+ " ?object <http://dbpedia.org/ontology/birthPlace> ?v0 . "

					// + " FILTER NOT EXISTS {?subject <http://dbpedia.org/ontology/birthPlace> ?otherv1. "
					// + " ?object <http://dbpedia.org/ontology/birthPlace> ?otherv1. } "
					+ " FILTER (?subject = <" + example.getLeft() + ">)"
					+ " FILTER (?object = <" + example.getRight() + ">)"
					+ "  }  ";
				if (get_pair(sparqlQuery) == 1)
					count_n = count_n + 1 ;
			
	        }

	        System.out.println("- Negative Coverage : " + count_n);

			int count_p = 0;
	        for (final  Pair<String,String> example  : positiveExamples) {
	        	
	        	sparqlQuery = "SELECT DISTINCT ?subject ?object  FROM <http://dbpedia.org>" 
					+ " WHERE { "
					+ " ?object <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person>.  "
					+ " ?subject <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person>. "

					+ " {{?subject ?targetRelation ?realObject.} UNION  {?realSubject ?targetRelation ?object.}}   "
					// + " ?subject ?otherRelation ?object.   "
					+ " FILTER (?targetRelation = <http://dbpedia.org/ontology/spouse>)   "
					// + " FILTER (?otherRelation = <http://dbpedia.org/ontology/spouse>)   "
					// + " FILTER NOT EXISTS {?subject <http://dbpedia.org/ontology/spouse> ?object.}  "
					+ " ?subject <http://dbpedia.org/ontology/birthPlace> ?v0 . "
					+ " ?object <http://dbpedia.org/ontology/birthPlace> ?v0 . "

					// + " FILTER NOT EXISTS {?subject <http://dbpedia.org/ontology/birthPlace> ?otherv1. "
					// + " ?object <http://dbpedia.org/ontology/birthPlace> ?otherv1. } "
					+ " FILTER (?subject = <" + example.getLeft() + ">)"
					+ " FILTER (?object = <" + example.getRight() + ">)"
					+ "  }  ";
				if (get_pair(sparqlQuery) == 1)
					count_p = count_p + 1 ;
			
	        }

	        System.out.println("- Positive Coverage : " + count_p);
			*/


	        // List<Set<String>> all_subTypes = new ArrayList(entity2types.values());

	        // Set<String> subjectTypes = all_subTypes.get(0);

	        // for (int i = 1; i < all_subTypes.size(); i++) {
	        //     subjectTypes.retainAll(all_subTypes.get(i));
	            
	        // }   

	        // System.out.println(" Subject Types : " + entity2types);

	        // entity2types = Maps.newHashMap();
	        // for (final String object : objectSet) {
	        //     try {
	        //         sparqlQuery = "SELECT ?o  FROM " + graph + " WHERE {" + object + type_prefix + " ?o }  ";
	        //         Set<String> currentTypes = get_type(sparqlQuery);
	        //         // entity2types.put(object, currentTypes);
	        //         for (final String type_ : currentTypes) 
	        //         	entity2types.put(type_, entity2types.getOrDefault(type_, 0) + 1);
	        //         if (currentTypes.contains("empty_result"))
	        //             System.out.println("Entity " + object.toString() + " has no types " );
	        //         // System.out.println("Entity " + object.toString() + " : " + currentTypes);
	        //     }
	        //     catch (final Exception e) {
	        //         System.out.println("Thread with entity " + object.toString() + " was not able to complet its job." + e);
	        //     }
	        // }

	        // List<Set<String>> all_objTypes = new ArrayList(entity2types.values());

	        // Set<String> objectTypes = all_objTypes.get(0);

	        // for (int i = 1; i < all_objTypes.size(); i++) {
	        //     objectTypes.retainAll(all_objTypes.get(i));
	            
	        // }   

			/*	        
	        System.out.println(" Generate Positive Examples ");
	        Set<Pair<String,String>> positiveExamples_v2;
	        Set<String> relations_v2= Sets.newHashSet();
	        relations_v2.add("http://dbpedia.org/ontology/birthYear");
			positiveExamples_v2 = naive.generatePositiveExamples(relations_v2, "http://dbpedia.org/ontology/Person", null, 500);
			for (final  Pair<String,String> example  :positiveExamples_v2) {
				// String[] tmpp = example.getRight().split("\\^\\^",2);
				// String tmppp = "\""+tmpp[0]+"\""+"^^"+"<"+tmpp[1]+">";
				System.out.println("Positive example: " + example.getLeft() + " --- " + example.getRight());
			}
			System.out.println(positiveExamples_v2.size());
	        System.out.println(" Generate Negative Examples ");
	        Set<Pair<String,String>> negativeExamples_v2;
			negativeExamples_v2 = naive.generateNegativeExamples(relations_v2, "http://dbpedia.org/ontology/Person", null, 500);
			for (final  Pair<String,String> example  :negativeExamples_v2) {
				// String[] tmpp = example.getRight().split("\\^\\^",2);
				// String tmppp = "\""+tmpp[0]+"\""+"^^"+"<"+tmpp[1]+">";
				System.out.println("Negative example: " + example.getLeft() + " --- " + example.getRight());
			}
			int count = 0;
			// for (final  Pair<String,String> example  :negativeExamples_v2) {
			// 	System.out.println(count);
			// 	count = count + 1;
			// 	String[] tmp_1 = example.getRight().split("\\^\\^",2);
   //        		String ex_right = "\""+tmp_1[0]+"\""+"^^"+"<"+tmp_1[1]+">";
			// 	 sparqlQuery = "SELECT ?p  FROM " + graph + " WHERE { ?s ?p ?o . FILTER (?s = <" + example.getLeft() + ">) " + 
   //      "FILTER (?o = " + ex_right + " )" + " }  ";
   //      	    Set<String> currentTypes = get_type(sparqlQuery);
	   
	  //       	for (final String type_ : currentTypes) 
			// 		System.out.println(" : " + type_);
			// }
			System.out.println(negativeExamples_v2.size());

			
		    // final List<HornRule> outputRules_positive = naive.discoverPositiveHornRules(negativeExamples, positiveExamples, relations, typeSubject, typeObject);
		    // final Map<HornRule, Double> outputRules_positive_v2 = naive.discoverAllPositiveHornRules(negativeExamples_v2, positiveExamples_v2, relations_v2, "http://dbpedia.org/ontology/birthYear", null, -1);
		    // List<HornRule> rule_positive_v2 = (outputRules_positive_v2 != null) ? Lists.newArrayList(outputRules_positive_v2.keySet()) : null;

			final Map<HornRule, Double> outputRules_negative_v2 = naive.discoverAllNegativeHornRules(negativeExamples_v2, positiveExamples_v2, relations_v2, "http://dbpedia.org/ontology/Person", null, -1);
		 	List<HornRule> rule_negative_v2 = (outputRules_negative_v2!= null) ? Lists.newArrayList(outputRules_negative_v2.keySet()) : null;
		    // final Instant endTime = Instant.now();
		    // LOGGER.info("----------------------------COMPUTATION ENDED----------------------------");
		    // // LOGGER.info("Final computation time: {} seconds.",(endTime.toEpochMilli()-startTime.toEpochMilli())/1000.);
		    // LOGGER.info("----------------------------Final positive output rules----------------------------");
		    // for(final HornRule oneRule:rule_positive_v2){
		    //   	LOGGER.info("{} - {}",oneRule, outputRules_positive_v2.get(oneRule));
		    // }
			
		     LOGGER.info("----------------------------COMPUTATION ENDED----------------------------");
		     // LOGGER.info("Final computation time: {} seconds.",(endTime.toEpochMilli()-startTime.toEpochMilli())/1000.);
		     LOGGER.info("----------------------------Final negative output rules----------------------------");
		     for(final HornRule oneRule:rule_negative_v2){
		       	LOGGER.info("{} - {}",oneRule,outputRules_negative_v2.get(oneRule));
		     }			
			 
			*/
			
		    load_rules_from_xml("tmp.xml", naive);

		    System.out.println("---------- EXTENDING RULES ------------------");
		    /*
		    Iterator<Pair<Set<RuleAtom>,Double>> iter = new_posrules.iterator();
		    while (iter.hasNext()) {
		    	Pair<Set<RuleAtom>,Double> rule_score = iter.next();
		    	Set<RuleAtom> rule = rule_score.getLeft();
		    	double conf_val = rule_score.getRight();
		    	double new_conf_val = naive.getRuleConfidence(rule, relations, typeSubject, typeObject, true);
		    	if (new_conf_val > conf_val) {
					extend_posrules.add(Pair.of(rule, new_conf_val));
		    	}

		    }

		    iter = new_negrules.iterator();
		    while (iter.hasNext()) {
		    	Pair<Set<RuleAtom>,Double> rule_score = iter.next();
		    	Set<RuleAtom> rule = rule_score.getLeft();
		    	double conf_val = rule_score.getRight();
		    	double new_conf_val = naive.getRuleConfidence(rule, relations, typeSubject, typeObject, false);
		    	if (new_conf_val > conf_val) {
					extend_negrules.add(Pair.of(rule, new_conf_val));
		    	}

		    }
			*/
		    System.out.println("----------Positive Result ------------------");
		    Iterator<Pair<Set<RuleAtom>,Double>> iter = current_posrules.iterator();
		    while (iter.hasNext()) {
		    	Pair<Set<RuleAtom>,Double> rule_score = iter.next();
		    	Set<RuleAtom> rule = rule_score.getLeft();
		    	double conf_val = rule_score.getRight();
		 		LOGGER.info("{} - {}",rule,conf_val);

		    }	

		    iter = extend_posrules.iterator();
		    while (iter.hasNext()) {
		    	Pair<Set<RuleAtom>,Double> rule_score = iter.next();
		    	Set<RuleAtom> rule = rule_score.getLeft();
		    	double conf_val = rule_score.getRight();
		 		LOGGER.info("{} - {}",rule,conf_val);

		    }

		    System.out.println("----------Negative Result ------------------");
		    iter = current_negrules.iterator();
		    while (iter.hasNext()) {
		    	Pair<Set<RuleAtom>,Double> rule_score = iter.next();
		    	Set<RuleAtom> rule = rule_score.getLeft();
		    	double conf_val = rule_score.getRight();
		 		LOGGER.info("{} - {}",rule,conf_val);

		    }	

		    iter = extend_negrules.iterator();
		    while (iter.hasNext()) {
		    	Pair<Set<RuleAtom>,Double> rule_score = iter.next();
		    	Set<RuleAtom> rule = rule_score.getLeft();
		    	double conf_val = rule_score.getRight();
		 		LOGGER.info("{} - {}",rule,conf_val);

		    }

		    // write_extended_rules("spouse", "tmp.xml", extend_posrules, extend_negrules);

	        System.exit(1);
		    	
	    } 
    	catch (FileNotFoundException e) {
        	System.out.println(" File not found !!");
        	System.exit(1);
    	}
 

       	final Instant startTime = Instant.now();
    

	    //compute outputs
	    // final List<HornRule> outputRules_negative = naive.discoverNegativeHornRules(negativeExamples, positiveExamples, relations, typeSubject, typeObject);

	    
	     final Map<HornRule, Double> outputRules_negative = naive.discoverAllNegativeHornRules(negativeExamples, positiveExamples, relations, typeSubject, typeObject, -1);
		 // List<HornRule> rule_negative = (outputRules_negative!= null) ? Lists.newArrayList(outputRules_negative.keySet()) : null;



	   
	    

	    // final List<HornRule> outputRules_positive = naive.discoverPositiveHornRules(negativeExamples, positiveExamples, relations, typeSubject, typeObject);
	    final Map<HornRule, Double> outputRules_positive = naive.discoverAllPositiveHornRules(negativeExamples, positiveExamples, relations, typeSubject, typeObject, -1);
	    // List<HornRule> rule_positive = (outputRules_positive != null) ? Lists.newArrayList(outputRules_positive.keySet()) : null;
	    // final Instant endTime = Instant.now();
	    
	    
	     // LOGGER.info("----------------------------COMPUTATION ENDED----------------------------");
	     // // LOGGER.info("Final computation time: {} seconds.",(endTime.toEpochMilli()-startTime.toEpochMilli())/1000.);
	     // LOGGER.info("----------------------------Final negative output rules----------------------------");
	     // for(final HornRule oneRule:rule_negative){
	     //   	LOGGER.info("{} - {}",oneRule,outputRules_negative.get(oneRule));
	     // }

	    // final Instant endTime = Instant.now();
	    // LOGGER.info("----------------------------COMPUTATION ENDED----------------------------");
	    // // LOGGER.info("Final computation time: {} seconds.",(endTime.toEpochMilli()-startTime.toEpochMilli())/1000.);
	    // LOGGER.info("----------------------------Final positive output rules----------------------------");
	    // for(final HornRule oneRule:rule_positive){
	    //   	LOGGER.info("{} - {}",oneRule, outputRules_positive.get(oneRule));
	    // }

	  //   LOGGER.info("----------------------------EXTEND RULE----------------------------"); 
	  //  	for(final HornRule oneRule:rule_positive){
	  //  		// MultipleGraphHornRule<String> newHornRule = oneRule.duplicateRule();
	  //  		RuleAtom newAtom = null;
	  //  		newAtom = new RuleAtom("subject", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://dbpedia.org/ontology/Royalty");
			// oneRule.addRuleAtom(newAtom, false);

	  //      	LOGGER.info("{} - {}",oneRule,naive.getRuleConfidence(oneRule, relations, typeSubject, typeObject, true));
	  //   }
   		
		write_xml("spouse", "tmp.xml", entity2types_sub, entity2types_obj, outputRules_positive, outputRules_negative,  (int) (positiveExamples.size() + negativeExamples.size())/10);

		
	}
		





	
}
