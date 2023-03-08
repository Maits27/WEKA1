package Praktika4;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import static weka.classifiers.lazy.IBk.*;

public class Praktika4 {

    public static void main(String[] args) {
        try{
            String dataPath, emaitzakPath;
            if(args.length==0){
                dataPath= "E:\\EHES\\WEKAPRUEBAS\\wine.arff";
                emaitzakPath ="E:\\EHES\\WEKAPRUEBAS\\4emaitzak.txt";
                ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataPath);
                Instances data = source.getDataSet();
                //data.setClassIndex(data.numAttributes()-1);
                data.setClassIndex(0);

                /*Klasifikadore honek 3 parametro ditu
                    k: auzokide kopurua (KNN);
                    d: metrika (nearestNeighbourSearchAlgorithm → distanceFunction );
                    w: distantziaren ponderazio faktorea (distanceWeighting)*/
                IBk ibk = new IBk();
                ibk.buildClassifier(data);

                /*  k parametroa lortzeko soilik for numeriko baten bidez egin dezakegu,
                    beraz ez da beharrezkoa ezer ezartzea

                    d aldiz, hurrengo distantzia neurtzeko algoritmoen artean hoberena
                    aukeratuz atera beharko dugu*/

                LinearNNSearch chebyshevDistance = new LinearNNSearch();
                chebyshevDistance.setDistanceFunction(new ChebyshevDistance());
                LinearNNSearch euclideanDistance = new LinearNNSearch();
                euclideanDistance.setDistanceFunction(new EuclideanDistance());
                LinearNNSearch manhattanDistance = new LinearNNSearch();
                manhattanDistance.setDistanceFunction(new ManhattanDistance());
                LinearNNSearch filteredDistance = new LinearNNSearch();
                filteredDistance.setDistanceFunction(new FilteredDistance());
                LinearNNSearch minkowskiDistance = new LinearNNSearch();
                minkowskiDistance.setDistanceFunction(new MinkowskiDistance());

                LinearNNSearch[] dist = new LinearNNSearch[] {chebyshevDistance, euclideanDistance,
                        manhattanDistance, filteredDistance, minkowskiDistance};

                /*  w etiketak dira -> Hauen arraya sortu. 3 etiketa posible daude WEKA-ren arera:
                    No distance weighting --> WEIGHT_NONE
                    Weight by 1/distance --> WEIGHT_INVERSE
                    Weight by 1-distance --> WEIGHT_SIMILARITY*/

                SelectedTag[] etiquetas = new SelectedTag[] {new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING),
                        new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING),
                        new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING)};

                //k, d eta w balioak gordetzen joateko:

                int kaux = 0;
                LinearNNSearch daux = null;
                SelectedTag waux = null;
                int iterazioa=0;
                // Iterazio bakoitzeko fmeasure (f) eta fmeasure maximo (fmax)

                double f = 0.0;
                double fmax = 0.0;
                 //Hold out erabiliz evaluatzailea bakarrik test multzoarekin probatu
                for (int k = 1; k<(data.numInstances()/4); k++){ //Auzokide kop laurdenerarte
                    ibk.setKNN(k);
                    for (LinearNNSearch d:dist){
                        ibk.setNearestNeighbourSearchAlgorithm(d);
                        for (SelectedTag w:etiquetas){
                            ibk.setDistanceWeighting(w);

                            // Ebaluatzailea sortu:
                            //TODO SI LO HACES CON HOLD OUT HACES BUILD CON TRAIN Y EVALUAS CON TEST
                            Evaluation eval = new Evaluation(data);
                            eval.crossValidateModel(ibk, data, 10, new Random(1));

                            f = eval.weightedFMeasure();
                            iterazioa++;
                            if (f > fmax){
                                fmax = f;
                                kaux = k;
                                daux = d;
                                waux = w;
                                System.out.println(iterazioa+" iterazioan lortu da hobekuntza, oraingo F-Measure: "+fmax);
                            }
                        }
                    }
                }

                System.out.println("   ");
                System.out.println("  ____________________________________________________________________________________________________ ");
                System.out.println("   ");
                System.out.println(iterazioa+" iterazio egin ostean: ");
                System.out.println(" Lortutako F-Measure maximoa: " + fmax);
                System.out.println("HURRENGO PARAMETROAK ERABILIKO DIRA:");
                System.out.println(" k = " + kaux);
                System.out.println(" d = " + daux.getDistanceFunction().getClass());
                System.out.println(" w = " + waux.getSelectedTag());
                //https://waikato.github.io/weka-wiki/optimizing_parameters/

                FileWriter fileWriter= new FileWriter(emaitzakPath);
                BufferedWriter bf = new BufferedWriter(fileWriter);
                bf.append(" Lortutako F-Measure maximoa: " + fmax);
                bf.append("\nHURRENGO PARAMETROAK ERABILIKO DIRA:\n");
                bf.append(" k = " + kaux);
                bf.append("\n d = " + daux.getDistanceFunction().getClass());
                bf.append("\n w = " + waux+"\n\n\n\n");

                bf.append("     ____________________________________________________________________________________________________    \n Ebaluazioa: \n");
                Randomize r = new Randomize();
                r.setRandomSeed(42);
                r.setInputFormat(data);
                data= Filter.useFilter(data, r);

                ibk.setDistanceWeighting(waux);
                ibk.setKNN(kaux);
                ibk.setNearestNeighbourSearchAlgorithm(daux);
                ibk.buildClassifier(data);

                Evaluation evaluation=new Evaluation(data);
                evaluation.crossValidateModel(ibk, data, 10, new Random());

                bf.append("\nLortutako emaitzen laburpena:\n"+evaluation.toSummaryString());
                bf.append("\nKlasearen neurriak:\n"+evaluation.toClassDetailsString());
                bf.append("\nNahasmen matrizea: \n"+evaluation.toMatrixString());
                bf.close();





                // ***************** CVParameterSelection erabiliz *********************************************************
                CVParameterSelection parameterSelection = new CVParameterSelection();
                parameterSelection.setClassifier(new IBk());
                parameterSelection.buildClassifier(data);
                //System.out.println("\n\n\n\n\n\n\nCVPARAMETER ERABILIZ: \n");
                //System.out.println("k: " + parameterSelection.getBestClassifierOptions()[1]);
                //System.out.println("w: " + parameterSelection.getBestClassifierOptions()[3]);
                //System.out.println("d: " + parameterSelection.getBestClassifierOptions()[5].split("\"")[1].split(" ")[0].split("\\.")[2]);


                // ******************* GridSerch erebiliz ******************************************************************
                //TODO


            }else{
                dataPath=args[0];
                emaitzakPath=args[1];
            }
        }catch (Exception e){System.out.println(e.toString());}
    }

}
