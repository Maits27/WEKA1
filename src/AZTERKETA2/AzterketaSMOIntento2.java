package AZTERKETA2;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

public class AzterketaSMOIntento2 {
    public static void main(String[] args) throws Exception{
        String dataPath, blindPath, modelPath, evalPath, predPath;
        if(args.length==0){
            System.out.println("EZ DUZU FORMATO ONA SARTU .JAR DEITZEAN: java -jar programa.jar /path/to/data.arff" +
                    " /path/to/testblind.arff /path/to/model.model /path/to/eval.txt /path/to/predictions.txt");
        }else{
            dataPath = args[0];
            blindPath = args[1];
            modelPath = args[2];
            evalPath = args[3];
            predPath = args[4];

            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            //DATU ANALISIA:
            for (int i =0; i<data.numAttributes(); i++){
                if (data.attribute(i).isNominal()){
                    System.out.println(i+". "+data.attribute(i).name()+" distinct balio kop: "+data.attributeStats(i).distinctCount);
                }
            }
            int minIndex = 0;
            int minMaiz=0;
            for (int i =0; i<data.classAttribute().numValues(); i++){
                int m=data.attributeStats(data.classIndex()).nominalCounts[i];
                System.out.println(i+". "+data.classAttribute().value(i)+" maiztasuna: "+m);
                if(minMaiz==0 && m!=0){
                    minIndex=i;
                    minMaiz=m;
                }else if (m!=0 && m<minMaiz){
                    minIndex=i;
                    minMaiz=m;
                }
            }

            //SAILKATZAILE OPTIMOA:
            PolyKernel pk = new PolyKernel();
            SMO smo = new SMO();
            smo.setKernel(pk);
            double expMax=1; //TODO ES DOUBLE!!!!!!!!!!!!!!!!!!!!!!!!!!!SINO VA MAL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            double fmax=0.0;
            for(int i =1; i<6; i++){
                pk.setExponent(i);
                smo.setKernel(pk);
                smo.buildClassifier(data);
                Evaluation evaluation = new Evaluation(data);
                evaluation.crossValidateModel(smo, data, 3, new Random(1));
                double f = evaluation.weightedFMeasure();
                System.out.println("Exponentearen balioa "+i+" izanda, honen fmeasure da: "+f);
                if(f>fmax){
                    expMax=i;
                    fmax=f;
                    System.out.println("Fmeasure maximo berria, exponente maximo berria: "+ i);
                }
            }

            pk.setExponent(expMax);
            smo.setKernel(pk);
            smo.buildClassifier(data);
            weka.core.SerializationHelper.write(modelPath, smo);

            //Ebaluazio ez zintzoa:
            FileWriter f = new FileWriter(evalPath);
            BufferedWriter bf = new BufferedWriter(f);

            Evaluation evaluation = new Evaluation(data);
            evaluation.evaluateModel(smo, data);

            bf.append("\n KLASE MINORITARIOA: " + data.classAttribute().value(minIndex)+
                    "\n   HONEKIKO PARAMETROAK:" +
                    "\n       PRECISION: " + evaluation.precision(minIndex)+
                    "\n       RECALL: " + evaluation.recall(minIndex)+
                    "\n       FMEASURE: "+ evaluation.fMeasure(minIndex));
            bf.append("\nNAHASMEN MATRIZEA:\n"+ evaluation.toMatrixString());
            bf.close();

            //Predikzioak:
            source = new ConverterUtils.DataSource(blindPath);
            Instances blind = source.getDataSet();
            blind.setClassIndex(blind.numAttributes()-1);

            Classifier k = (Classifier) weka.core.SerializationHelper.read(modelPath);

            f = new FileWriter(evalPath);
            bf = new BufferedWriter(f);

            for(int i =1; i<blind.numInstances(); i++){
                bf.append(i+" instantzia ---> "+k.classifyInstance(blind.instance(i)));
            }
            bf.close();
        }
    }
}
