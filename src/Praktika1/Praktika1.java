package Praktika1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Enumeration;

public class Praktika1 {
    public static void main(String[] args) {
        try{
            /*Aurrebaldintzak:
                1. argumentuan .arff fitxategi baten path-a hartzen da.
                Fitxategi horren klasea azken atributuan dator. */
            if(args.length==0){
                System.out.println("Sartu zuzen komandoa hurrengo formatua erabiliz: \n" +
                        "java -jar praktika1.jar /path/to/data.arff /path/to/emaitzak.txt");
            }else{
                String path=args[0];
                String path2=args[1];

                //Artxiboa kargatu:
                DataSource source= new DataSource(path);
                Instances data=source.getDataSet();

                data.setClassIndex(data.numAttributes() - 1);

                FileWriter file = new FileWriter(path2);
                BufferedWriter bf =new BufferedWriter(file);

                bf.append("Fitxategiaren path-a: "+ path); bf.newLine();
                bf.append("Fitxategiak instantzia kopuru hau ditu: "+data.numInstances()); bf.newLine();
                bf.append("Fitxategiak atributu kopuru hau ditu: "+data.numAttributes()+"\n"); bf.newLine();
                bf.append(data.attribute(0).name()+", lehen atributuak, honako balio kopuru hau har ditzake (distinct): "
                        + data.numDistinctValues(data.attribute(0))+"\n"); bf.newLine();

                //Klase atributuaren informazioa inprimatzeko:
                bf.append(data.attribute(data.numAttributes()-1).name()+
                        ", azken atributuak, honako balio hauek har ditzake: \n");
                //Hartu ditzaeen atributu balioak enumeratu
                Enumeration<Object> balioak=data.classAttribute().enumerateValues();
                int kont=0;
                int gutxien=0;
                //Banan banan errekorritu elementuak "gutxien" atributuan maiztasun txikiena gordez
                while(balioak.hasMoreElements()){
                    Object n=balioak.nextElement();
                    int maiztasuna= data.attributeStats(data.classIndex()).nominalCounts[kont++];

                    if(kont==1){gutxien=maiztasuna;}
                    else if(maiztasuna<gutxien){gutxien=maiztasuna;}

                    bf.append("     "+n+" balioa, maiztasun honekin atera da: "+maiztasuna+"\n");
                }
                bf.newLine();

                //Maiztasun txikiena duten elementuak inprimatu:
                bf.append("Gutxien atera diren atributuak ("+gutxien+" aldiz), hau da, klase minoritarioak: ");
                int k=0;
                balioak=data.classAttribute().enumerateValues();
                while (balioak.hasMoreElements()){
                    Object n=balioak.nextElement();
                    if(data.attributeStats(data.classIndex()).nominalCounts[k++]==gutxien){bf.append("     "+n+"\n");}
                }

                bf.newLine();
                bf.append(data.attribute(data.classIndex()-1).name()+
                        ", azken aurreko atributuak, honako missing balio kopuru hau ditu: "
                        +data.attributeStats(data.classIndex()-1).missingCount+"\n");

                //1 PRAKTIKA GIDOIAREN 7. GALDERA (Lehen 5 atributuen informazioa atera)
                bf.append("\nLehen 5 atributuen informazioa: \n");
                for(int i=0; i<5; i++) {
                    bf.append("Atributua: " + data.attribute(i).name()+"\n");

                    //Zein motakoa den ateratzeko:
                    boolean numerikoa = false;
                    if (data.attribute(i).isNominal()) {
                        bf.append("     Nominala da.\n");
                    } else if (data.attribute(i).isString()) {
                        bf.append("     String da.\n");
                    } else if (data.attribute(i).isDate()) {
                        bf.append("     Data da.\n");
                    } else {
                        bf.append("     Numerikoa da.\n");
                        numerikoa = true;
                    }

                    //Mota kontuan izan barik missing, distinct eta unique balioak dauden jakiteko:
                    bf.append("     Missing balio kopurua: " + data.attributeStats(i).missingCount+"\n");
                    bf.append("     Distinct balio kopurua: " + data.attributeStats(i).distinctCount+"\n");
                    bf.append("     Unique balio kopurua: " + data.attributeStats(i).uniqueCount+"\n");

                    //Numerikoa bada atera beharreko datuak:
                    if (numerikoa) {
                        bf.append("     Min: " + data.attributeStats(i).numericStats.min+"\n");
                        bf.append("     Max: " + data.attributeStats(i).numericStats.max+"\n");
                        bf.append("     Batazbeste: " + data.attributeStats(i).numericStats.mean+"\n");
                        bf.append("     Desbiderapen: " + data.attributeStats(i).numericStats.stdDev+"\n");
                    }
                    bf.newLine();
                }
                bf.close();
            }
        }catch (Exception e){System.out.println("Error: "+e.toString());}
    }
}