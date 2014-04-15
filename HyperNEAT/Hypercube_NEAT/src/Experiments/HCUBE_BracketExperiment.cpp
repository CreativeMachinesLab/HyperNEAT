#include "MarchCube.h"
#include <iomanip> 
#include "Experiments/HCUBE_BracketExperiment.h"
#include <stdlib.h>
#include <time.h>
//#include "DigitalMatter.h"
//#include "DM_FEA.h" 
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctype.h>
//ACHEN: Included for temporary hack
#include <tinyxml.h> // #include “tinyxml.h"?
#include <string>
#ifdef VISUALIZESHAPES
//#include <SFML/Graphics.hpp>
#endif
#include <sstream>


#define SHAPES_EXPERIMENT_DEBUG (0)

namespace HCUBE
{
    using namespace NEAT;

    BracketExperiment::BracketExperiment(string _experimentName)
    :   Experiment(_experimentName)
    {
        
        cout << "Constructing experiment named BracketExperiment.\n";

		addDistanceFromCenter = true;
		addDistanceFromCenterXY = true;
		addDistanceFromCenterYZ = true;
		addDistanceFromCenterXZ = true;
			
		convergence_step = int(NEAT::Globals::getSingleton()->getParameterValue("ConvergenceStep"));
		if(convergence_step < 0)
			convergence_step = INT_MAX; // never switch
/*
		// initialize size/granularity of workspace 
		WorkSpace = Vec3D(0.01, 0.02, 0.01); //reasonable workspace (meters)
//		VoxelSize = 0.005; //Size of voxel in meters (IE the lattice dimension); (original)
		VoxelSize = 0.001; //Size of voxel in meters (IE the lattice dimension); (what all the paper screen shots used)
//		VoxelSize = 0.0005; //Size of voxel in meters (IE the lattice dimension); (original)
		voxelExistsThreshold = 0.0;//@# MIGHT CHANGE ACHEN
		num_x_voxels = int(WorkSpace.x/VoxelSize);
		num_y_voxels = int(WorkSpace.y/VoxelSize);
		num_z_voxels = int(WorkSpace.z/VoxelSize);
*/

		num_x_voxels = (int)NEAT::Globals::getSingleton()->getParameterValue("BoundingBoxLength");
		num_y_voxels = num_x_voxels;
		num_z_voxels = num_x_voxels;
		cout << "NUM X VOXELS: " << num_x_voxels << endl;

		int FixedBoundingBoxSize = (int)NEAT::Globals::getSingleton()->getParameterValue("FixedBoundingBoxSize");
		if (FixedBoundingBoxSize > 0)
		{
			VoxelSize = FixedBoundingBoxSize*0.001/num_x_voxels;
		} else {
			VoxelSize = 0.001;
		}

		// nac: set the bounding box size to that used for 5^3 (try 10^3 too?)
		//VoxelSize = 5.0*0.001/num_x_voxels;

		cout << "Voxel Size: " << VoxelSize << endl;
		cout << "Total Bounding Box Size: " << VoxelSize * num_x_voxels << endl;

		#ifndef INTERACTIVELYEVOLVINGSHAPES		
		//TargetContinuousArray = CArray3Df(num_x_voxels, num_y_voxels, num_z_voxels);	//because calling the consrucor on TargetContinuousArray only initialized it locally (!?!)u
		//generateTarget3DObject(); // fill array with target shape
		#endif
        
    }

    NEAT::GeneticPopulation* BracketExperiment::createInitialPopulation(int populationSize)
    {

        NEAT::GeneticPopulation* population = new NEAT::GeneticPopulation();

        vector<GeneticNodeGene> genes;
	cout << "JMC HERE" << endl;

        genes.push_back(GeneticNodeGene("Bias","NetworkSensor",0,false));
        genes.push_back(GeneticNodeGene("x","NetworkSensor",0,false));
        genes.push_back(GeneticNodeGene("y","NetworkSensor",0,false));
        genes.push_back(GeneticNodeGene("z","NetworkSensor",0,false));
		if(addDistanceFromCenter)	genes.push_back(GeneticNodeGene("d","NetworkSensor",0,false));
		if(addDistanceFromCenterXY) genes.push_back(GeneticNodeGene("dxy","NetworkSensor",0,false));
		if(addDistanceFromCenterYZ) genes.push_back(GeneticNodeGene("dyz","NetworkSensor",0,false));
		if(addDistanceFromCenterXZ) genes.push_back(GeneticNodeGene("dxz","NetworkSensor",0,false));
		genes.push_back(GeneticNodeGene("OutputEmpty","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (numMaterials > 3 ) genes.push_back(GeneticNodeGene("OutputPassiveStiff","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (numMaterials > 1 ) genes.push_back(GeneticNodeGene("OutputPassiveSoft","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (numMaterials > 0 ) genes.push_back(GeneticNodeGene("OutputActivePhase1","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (numMaterials > 2 ) genes.push_back(GeneticNodeGene("OutputActivePhase2","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
        for (size_t a=0;a<populationSize;a++)
        {
            shared_ptr<GeneticIndividual> individual(new GeneticIndividual(genes,true,1.0));

            //for (size_t b=0;b<0;b++)  // this was uncommented in jason's original code, but I think since he has b<0 it never fired. setting it to >0 would probably testMutate it
            //{            
            //    individual->testMutate();
            //}

            population->addIndividual(individual);
        }

        return population;
    }


    double BracketExperiment::mapXYvalToNormalizedGridCoord(const int & r_xyVal, const int & r_numVoxelsXorY) 
    {
        // turn the xth or yth node into its coordinates on a grid from -1 to 1, e.g. x values (1,2,3,4,5) become (-1, -.5 , 0, .5, 1)
        // this works with even numbers, and for x or y grids only 1 wide/tall, which was not the case for the original
        // e.g. see findCluster for the orignal versions where it was not a funciton and did not work with odd or 1 tall/wide #s
		
        double coord;
                
        if(r_numVoxelsXorY==1) coord = 0;
        else                  coord = -1 + ( r_xyVal * 2.0/(r_numVoxelsXorY-1) );

        return(coord);    
    }
	

    double BracketExperiment::processEvaluation(shared_ptr<NEAT::GeneticIndividual> individual, string individualID, int genNum, bool saveVxaOnly, double bestFit)
    {
		//initializes continuous space array with zeros. +1 is because we need to sample
		// these points at all corners of each voxel, leading to n+1 points in any dimension
		CArray3Df ContinuousArray(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df PassiveStiffArray(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df PassiveSoftArray(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df Phase1Array(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df Phase2Array(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array


		NEAT::FastNetwork <double> network = individual->spawnFastPhenotypeStack<double>();        //JMC: this is the CPPN network

		int px, py, pz; //temporary variable to store locations as we iterate
		float xNormalized;
		float yNormalized;
		float zNormalized;
		float distanceFromCenter;
		float distanceFromCenterXY;
		float distanceFromCenterYZ;
		float distanceFromCenterXZ;
		int ai;
	
		for (int j=0; j<ContinuousArray.GetFullSize(); j++)  //iterate through each location of the continuous matrix
		{ 
			ContinuousArray.GetXYZ(&px, &py, &pz, j); //gets XYZ location of this element in the array
			xNormalized = mapXYvalToNormalizedGridCoord(px, num_x_voxels);
			yNormalized = mapXYvalToNormalizedGridCoord(py, num_y_voxels);
			zNormalized = mapXYvalToNormalizedGridCoord(pz, num_z_voxels);
			
			//calculate input vars
			if(addDistanceFromCenter)   distanceFromCenter   = sqrt(pow(double(xNormalized),2.0)+pow(double(yNormalized),2.0)+pow(double(zNormalized),2.0));			
			if(addDistanceFromCenterXY) distanceFromCenterXY = sqrt(pow(double(xNormalized),2.0)+pow(double(yNormalized),2.0));
			if(addDistanceFromCenterYZ) distanceFromCenterYZ = sqrt(pow(double(yNormalized),2.0)+pow(double(zNormalized),2.0));
			if(addDistanceFromCenterXZ) distanceFromCenterXZ = sqrt(pow(double(xNormalized),2.0)+pow(double(zNormalized),2.0));				
	
			network.reinitialize();								//reset CPPN
			network.setValue("x",xNormalized);					//set the input numbers
			network.setValue("y",yNormalized);
			network.setValue("z",zNormalized);
			if(addDistanceFromCenter) network.setValue("d",distanceFromCenter);
			if(addDistanceFromCenterXY) network.setValue("dxy",distanceFromCenterXY);
			if(addDistanceFromCenterYZ) network.setValue("dyz",distanceFromCenterYZ);
			if(addDistanceFromCenterXZ) network.setValue("dxz",distanceFromCenterXZ);
			//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
			//if(addInsideOrOutside) network.setValue("inout",e);
			network.setValue("Bias",0.3);                       
							
			network.update();                                   //JMC: on this line we run the CPPN network...  
			
			ContinuousArray[j] = network.getValue("OutputEmpty");        //JMC: and here we get the CPPN output (which is the weight of the connection between the two)
			if (numMaterials > 3 )PassiveStiffArray[j] = network.getValue("OutputPassiveStiff");
			if (numMaterials > 1 )PassiveSoftArray[j] = network.getValue("OutputPassiveSoft");
			if (numMaterials > 0 ) Phase1Array[j] = network.getValue("OutputActivePhase1");
			if (numMaterials > 2 )Phase2Array[j] = network.getValue("OutputActivePhase2");
		}
	
		float fitness;
		float origFitness = 0.001;
		int voxelPenalty = writeVoxelyzeFile(ContinuousArray, PassiveStiffArray, PassiveSoftArray, Phase1Array, Phase2Array, individualID);
		
//		if (saveVxaOnly)
//		{
//			std::ostringstream addFitnessToInputFileCmd;
//			addFitnessToInputFileCmd << "mv " << individualID << "_genome.vxa "<< genNum << "/" << individualID << "--adjFit_"<< individual.getFitness() << "--numFilled_" << individual.getNumFilled() << "--origFit_"<< individual.getOrigFitness() << "_genome.vxa";
//			int exitCode3 = std::system(addFitnessToInputFileCmd.str().c_str());
//			return 0.001;
//		}

		if (voxelPenalty < 2)
		{
			fitness = 0.0001;
			cout << "individual had less than 2 voxels filled, skipping evaluation..." << endl;
			//skippedEvals++;	
		} else
		{
			//nac: check md5sum of vxa file	
			//std::ostringstream md5sumCmd;
			//md5sumCmd << "md5sum " << individualID << "_genome.vxa";
			//FILE* pipe = popen(md5sumCmd.str().c_str(), "r");

			FILE* pipe = popen("md5sum md5sumTMP.txt", "r");			
			if (!pipe) {cout << "ERROR 1, exiting." << endl; exit(1);}
			char buffer[128];
			std::string result = "";
			while(!feof(pipe)) 
			{
				if(fgets(buffer, 128, pipe) != NULL)
					result += buffer;
			}
			pclose(pipe);

			std::string md5sumString = result.substr(0,32);
			cout << "md5sum: " << md5sumString << endl;
			
			if (fitnessLookup.find(md5sumString) != fitnessLookup.end())
			{
				//fitness = fitnessLookup[md5sumString];
				//FitnessRecord fits = fitnessLookup[md5sumString];
				pair<double, double> fits = fitnessLookup[md5sumString];
				fitness = fits.second;
				//origFitness = fits.second;
				//cout << "fitness from hash: " << fitness << endl;
				//cout << "orig fitness: " << origFitness << endl;
				cout << "This individual was already evaluated!  Its original fitness look-up is: " << fitness << endl;

				//skippedEvals++;	

				//std::ostringstream rmVxaCmd;
				//rmVxaCmd << "rm " << individualID << "_genome.vxa";
				//int exitCode3 = std::system(rmVxaCmd.str().c_str());
/*
				std::ostringstream rmVxaCmd;

				if (genNum % (int)NEAT::Globals::getSingleton()->getParameterValue("RecordEntireGenEvery")==0) // || bestFit<fitness) // <-- nac: individuals processed one at a time, not as generation group
				{		
					rmVxaCmd << "mv " << individualID << "_genome.vxa Gen_"; 

					char buffer1 [100];
					sprintf(buffer1, "%04i", genNum);
					rmVxaCmd << buffer1;

					rmVxaCmd << "/" << individualID << "--adjFit_";

					char adjFitBuffer[100];
					sprintf(adjFitBuffer, "%.8lf", fitness);
					rmVxaCmd << adjFitBuffer;

					rmVxaCmd << "--numFilled_";

					char NumFilledBuffer[100];
					sprintf(NumFilledBuffer, "%04i", voxelPenalty);
					rmVxaCmd << NumFilledBuffer;

					//origFitness = individual->getOrigFitness();
					cout << "GRABBING ORIGINAL FITNESS: " << origFitness << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;

					rmVxaCmd << "--origFit_";

					char origFitBuffer[100];
					sprintf(origFitBuffer, "%.8lf", origFitness);
					rmVxaCmd << origFitBuffer;
					//rmVxaCmd << "hashed.fit";

					rmVxaCmd << "_genome.vxa";

				} else
				{
					rmVxaCmd << "rm " << individualID << "_genome.vxa";
				}

				int exitCode3 = std::system(rmVxaCmd.str().c_str());
*/
				//return fitness;
			} else 

			{
				timeval tim;
				gettimeofday(&tim, NULL);
				double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	
				//cout << "individual: " << individualID << endl;
		
				cout << "starting voxelyze evaluation now" << endl;

				std::ostringstream callVoxleyzeCmd;
				callVoxleyzeCmd << "./voxelize -f " << individualID << "_genome.vxa";
		
				int exitCode = std::system(callVoxleyzeCmd.str().c_str());

				//int exitCode2 = std::system("mv BracketOutputTmp.xml BracketOutput.xml");


				gettimeofday(&tim, NULL);
				double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
		
				printf("voxelyze took %.6lf seconds\n", t2-t1);

				//cout<<"wrote the file and exiting!" << endl;

				//timespec time1, time2;

				//float startClock = clock_gettime(CLOCK_MONOTONIC, &time1);

		
				//float timeUsed = clock_gettime(CLOCK_MONOTONIC, &time2) - startClock;

				//cout << "Voxelyze used " << timeUsed << " seconds" << endl;

				cout << "Exit Code From Voxelyze:" << WEXITSTATUS(exitCode) << endl;

				TiXmlDocument doc;
				std::ostringstream outFileName;
				outFileName << individualID << "_fitness.xml";   		// <-- for real
				//outFileName << "nan_fitness.xml";						// <-- debug only

				if(!doc.LoadFile(outFileName.str().c_str()))
				{
					cerr << doc.ErrorDesc() << endl;
					return 0.000001; //return FAILURE;
				}	
		
				TiXmlElement* root = doc.FirstChildElement();
				if(root == NULL)
				{
					cerr << "Failed to load file: No root element."
						 << endl;
					doc.Clear();
					return 0.001; //return FAILURE;
				}

				fitness = 0.001;

				for(TiXmlElement* elem1 = root->FirstChildElement(); elem1 != NULL; elem1 = elem1->NextSiblingElement())
				{
			
					for(TiXmlElement* elem = elem1->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
					{

						string elemName = elem->Value();
		/*				cout << "Elem Name: " << elemName << endl;
	
						const char* attr;
						if(elemName == "Fitness")//"FinalCOM_Dist")
						{
							//cout << "got here 4" << endl;
							attr = elem->Attribute("FinalCOM_Dist");
							if(attr != NULL)
							//cout<< "got here 3" << endl	; // Do stuff with it
							
						}
		*/
			
						for(TiXmlNode* e = elem->FirstChild(); e != NULL; e = e->NextSibling())
						{

							cout << "got here 1" << endl;
							TiXmlText* text = e->ToText();
							cout << "element from voxelyze: " << e->Value() << endl;
							cout << "raw fitness value from voxelyze: " << fabs(atof(e->Value())) << endl;
							cout << "body length fitness value from voxelyze: " << fabs(atof(e->Value()))/(num_x_voxels*VoxelSize) << endl;
							fitness = fabs(atof(e->Value()))/(num_x_voxels*VoxelSize);
							/*
							if(text != NULL)
							{
								cout << "text test 1: " << text->Value() << endl;
								//cout << "got here 5, text = NULL" << endl;
								//continue;
							//}
						*/
							if (std::string::npos != string(e->Value()).find('n'))
							{
							  fitness = 0.0001;
							  cout << "THIS INDIVIDUAL RETURNED NaN... setting arbitrary low fitness" << endl;
							}
						/*
							string t = text->Value();
							// Do stuff
							fitness = fabs(atof(t.c_str()))/(num_x_voxels*VoxelSize); 
							cout << "TEST FITNESS FROM ATOF: " << fitness << endl;
									//NOTE: taking abs of dispalcement (COM) for fitness (nac)
									//NOTE: using num body length in x direction as fitness (voxel size set at 0.001) (nac)
							cout << "got here 2" << endl;
							cout << "text = " << t << endl;

							}
							*/
						}
					}
				}
				doc.Clear();
			}

			
			if (fitness < 0.0001) fitness = 0.0001;
			if (fitness > 1000) fitness = 0.0001;

			cout << "Original fitness value (in body lengths): " << fitness << endl;
			int outOf;
			if (int(NEAT::Globals::getSingleton()->getParameterValue("PenaltyType")) == 3) 
			{
				// note: only works for cubes, foil out everything to get expression for rectangles
				outOf = (num_x_voxels*num_x_voxels*num_x_voxels - num_x_voxels*num_x_voxels)*3;
			} else {
				outOf = num_x_voxels*num_x_voxels*num_x_voxels;
			}
			cout << "Voxels (or Connections) per Possible Maximum: " << voxelPenalty << " / " << outOf << endl;  //nac: put in connection cost variant
			double voxelPenaltyf = 1.0 - pow(1.0*voxelPenalty/(outOf*1.0),NEAT::Globals::getSingleton()->getParameterValue("PenaltyExp"));
			//cout << "PenaltyExp: " << NEAT::Globals::getSingleton()->getParameterValue("PenaltyExp") << endl;
			//cout << "base: " << 1.0*voxelPenalty/(outOf*1.0) << endl;
			//cout << "pow: " << pow(1.0*voxelPenalty/(outOf*1.0),NEAT::Globals::getSingleton()->getParameterValue("PenaltyExp")) << endl;
			origFitness = fitness;
			if (int(NEAT::Globals::getSingleton()->getParameterValue("PenaltyType")) != 0) {fitness = fitness * (voxelPenaltyf);}
			//cout << "Fitness Penalty Adjustment: " << voxelPenaltyf << endl;
			if (int(NEAT::Globals::getSingleton()->getParameterValue("PenaltyType")) != 0) {printf("Fitness Penalty Multiplier: %f\n", voxelPenaltyf);}
			else {cout << "No penalty funtion, using original fitness" << endl;}

			// nac: update fitness lookup table		
			//FitnessRecord fitness, origFitness;
			//std::map<double, FitnessRecord> fits;

			if (fitness < 0.0001) fitness = 0.0001;
			if (fitness > 1000) fitness = 0.0001;

			pair <double, double> fits (fitness, origFitness);	
			fitnessLookup[md5sumString]=fits;
	
		}
		
		cout << "ADJUSTED FITNESS VALUE: " << fitness << endl;

		std::ostringstream addFitnessToInputFileCmd;
		std::ostringstream addFitnessToOutputFileCmd;

		if (genNum % (int)NEAT::Globals::getSingleton()->getParameterValue("RecordEntireGenEvery")==0) // || bestFit<fitness) // <-- nac: individuals processed one at a time, not as generation group
		{		
			addFitnessToInputFileCmd << "mv " << individualID << "_genome.vxa Gen_"; 

			char buffer1 [100];
			sprintf(buffer1, "%04i", genNum);
			addFitnessToInputFileCmd << buffer1;

			addFitnessToInputFileCmd << "/" << individualID << "--adjFit_";

			char adjFitBuffer[100];
			sprintf(adjFitBuffer, "%.8lf", fitness);
			addFitnessToInputFileCmd << adjFitBuffer;

			addFitnessToInputFileCmd << "--numFilled_";

			char NumFilledBuffer[100];
			sprintf(NumFilledBuffer, "%04i", voxelPenalty);
			addFitnessToInputFileCmd << NumFilledBuffer;

			addFitnessToInputFileCmd << "--origFit_";

			char origFitBuffer[100];
			sprintf(origFitBuffer, "%.8lf", origFitness);
			addFitnessToInputFileCmd << origFitBuffer;

			addFitnessToInputFileCmd << "_genome.vxa";

		} else
		{
			addFitnessToInputFileCmd << "rm " << individualID << "_genome.vxa";
		}

		//addFitnessToOutputFileCmd << "mv " << individualID << "_fitness.xml " << individualID << "--Fit_"<< fitness <<"_fitness.xml";
		addFitnessToOutputFileCmd << "rm -f " << individualID << "_fitness.xml";// << individualID << "--Fit_"<< fitness <<"_fitness.xml";

		int exitCode3 = std::system(addFitnessToInputFileCmd.str().c_str());
		int exitCode4 = std::system(addFitnessToOutputFileCmd.str().c_str());  
/*
		if (isalpha(fitness))
		{
			float fitness = 0;
			cout << "MAPPED FITNESS TO ZERO!!!" << endl;
		}
*/
		//exit(3);

		//int fitness = -999; //TODO: implement fitness evaluation
		
		if (fitness < 0.0001) fitness = 0.0001;	
		individual->setOrigFitness(origFitness);
		
			
		//individual.setOrigFitness(origFitness);     // nac: WHY DON'T THESE WORK??!!	
		//individual.setNumFilled(voxelPenalty);	
		
		return fitness; 
		//return 100;

    }
	

    void BracketExperiment::processGroup(shared_ptr<NEAT::GeneticGeneration> generation)
    {
		double bestFit = 0.0;

		for(int z=0;z< group.size();z++)
		{
			numVoxelsFilled   = 0;
			numVoxelsActuated = 0;
			numConnections    = 0;	

			//totalEvals++;

			//cout << "indivudial num??? ('z' value): " << z <<endl;

			numMaterials = int(NEAT::Globals::getSingleton()->getParameterValue("NumMaterials"));

			int genNum = generation->getGenerationNumber() + 1;

			char buffer2 [50];
			sprintf(buffer2, "%04i", genNum);

			std::ostringstream mkGenDir;
			if (genNum % (int)NEAT::Globals::getSingleton()->getParameterValue("RecordEntireGenEvery")==0)
			{
				mkGenDir << "mkdir -p Gen_" << buffer2;
			}			
	
			int exitCode5 = std::system(mkGenDir.str().c_str());

			shared_ptr<NEAT::GeneticIndividual> individual = group[z];

			//cout<< "RUN NAME: " << NEAT::Globals::getSingleton()->getOutputFilePrefix() << endl;

			std::ostringstream tmp;
			tmp << "Bracket--" << NEAT::Globals::getSingleton()->getOutputFilePrefix() << "--Gen_" ;
			char buffer [50];
			sprintf(buffer, "%04i", genNum);
			tmp << buffer;
			tmp << "--Ind_" << individual;
	  		string individualID = tmp.str();
			cout << individualID << endl;
			

			#if SHAPES_EXPERIMENT_DEBUG
			  cout << "in BracketExperiment.cpp::processIndividual, processing individual:  " << individual << endl;
			#endif
			double fitness = 0;
			
			
			fitness = processEvaluation(individual, individualID, genNum, false, bestFit);
			
			if (bestFit<fitness) bestFit = fitness;
					
			if(fitness > std::numeric_limits<double>::max())
			{
				cout << "error: the max fitness is greater than the size of the double. " << endl;
				cout << "max double size is: : " << std::numeric_limits<double>::max() << endl;
				exit(88);
			} 
			if (fitness < 0)
			{
				cout << "Fitness Less Than Zero!!!, it is: " << fitness << "\n";  
				exit(10);
			}
			
			#if SHAPES_EXPERIMENT_DEBUG        
				cout << "Individual Evaluation complete!\n";
				printf("fitness: %f\n", fitness);	
			#endif

			cout << "Individual Evaluation complete!\n";
			cout << "fitness: " << fitness << endl;
	  
			individual->reward(fitness);

			if (false)	//to print cppn
			{
				printNetworkCPPN(individual);    
			}
			
			//cout << "Total Evaluations: " << totalEvals << endl;
			//cout << "Skipped Evaluations: " << skippedEvals << endl;
			//cout << "(Total for run) PercentOfEvaluationsSkipped: " << double(totalEvals)/double(skippedEvals) << endl;
		}
    }

    void BracketExperiment::processIndividualPostHoc(shared_ptr<NEAT::GeneticIndividual> individual)
    {
		
		
        /*{
            mutex::scoped_lock scoped_lock(*Globals::ioMutex);
            cout << "Starting Evaluation on object:" << this << endl;
            cout << "Running on individual " << individual << endl;
        }

        cout << "Sorry, this was never coded up for ShapeRecognitioV1. You'll have to do that now." << endl;
        exit(6);
        
        //in jason's code, the got 10 points just for entering the game, wahooo!
        individual->setFitness(0);

        double fitness=0;

        double maxFitness = 0;

        //bool solved=true; @jmc: was uncommented in the original and produced an error forn not being used

        //cout << "Individual Evaluation complete!\n";

        //cout << maxFitness << endl;

        individual->reward(fitness);

        if (fitness >= maxFitness*.95)
        {
            cout << "PROBLEM DOMAIN SOLVED!!!\n";
        }
		*/
    }

    
    void BracketExperiment::printNetworkCPPN(shared_ptr<const NEAT::GeneticIndividual> individual)
    {
      cout << "Printing cppn network" << endl;
      ofstream network_file;        
      network_file.open ("networkCPPN-ThatSolvedTheProblem.txt", ios::trunc );
      
      NEAT::FastNetwork <double> network = individual->spawnFastPhenotypeStack <double>();        //JMC: this creates the network CPPN associated with this individual used to produce the substrate (neural network)

      network_file << "num links:" << network.getLinkCount() << endl;
      network_file << "num nodes:" << network.getNodeCount() << endl;
 
      int numLinks = network.getLinkCount();
      int numNodes = network.getNodeCount();
      ActivationFunction activationFunction;

      //print out which node corresponds to which integer (e.g. so you can translate a fromNode of 1 to "x1"  
      map<string,int> localNodeNameToIndex = *network.getNodeNameToIndex();
      for( map<string,int>::iterator iter = localNodeNameToIndex.begin(); iter != localNodeNameToIndex.end(); iter++ ) {
        network_file << (*iter).first << " is node number: " << (*iter).second << endl;
      }
      
      for (size_t a=0;a<numLinks;a++)
      {
          
          NetworkIndexedLink <double> link = *network.getLink(a);
           
          network_file << link.fromNode << "->" << link.toNode << " : " << link.weight << endl;
      }
      for (size_t a=0;a<numNodes;a++)
      {          
          activationFunction = *network.getActivationFunction(a);           
          network_file << " activation function " << a << ": ";
          if(activationFunction == ACTIVATION_FUNCTION_SIGMOID) network_file << "ACTIVATION_FUNCTION_SIGMOID";
          if(activationFunction == ACTIVATION_FUNCTION_SIN) network_file << "ACTIVATION_FUNCTION_SIN";
          if(activationFunction == ACTIVATION_FUNCTION_COS) network_file << "ACTIVATION_FUNCTION_COS";
          if(activationFunction == ACTIVATION_FUNCTION_GAUSSIAN) network_file << "ACTIVATION_FUNCTION_GAUSSIAN";
          if(activationFunction == ACTIVATION_FUNCTION_SQUARE) network_file << "ACTIVATION_FUNCTION_SQUARE";
          if(activationFunction == ACTIVATION_FUNCTION_ABS_ROOT) network_file << "ACTIVATION_FUNCTION_ABS_ROOT";
          if(activationFunction == ACTIVATION_FUNCTION_LINEAR) network_file << "ACTIVATION_FUNCTION_LINEAR";
          if(activationFunction == ACTIVATION_FUNCTION_ONES_COMPLIMENT) network_file << "ACTIVATION_FUNCTION_ONES_COMPLIMENT";
          if(activationFunction == ACTIVATION_FUNCTION_END) network_file << "ACTIVATION_FUNCTION_END";
          network_file << endl;
      }

      network_file.close();

      return;
    }

    Experiment* BracketExperiment::clone()
    {
        BracketExperiment* experiment = new BracketExperiment(*this);

        return experiment;
    }
    
	
	bool BracketExperiment::converged(int generation) {
		if(generation == convergence_step)
			return true;
		return false;
	}		

	void BracketExperiment::writeAndTeeUpParentsOrderFile() {		//this lets django know what the order in the vector was of each parent for each org in the current gen
		string outFilePrefix = NEAT::Globals::getSingleton()->getOutputFilePrefix();
		std::ostringstream parentFilename;
		std::ostringstream parentFilenameTemp;
		std::ostringstream parentFilenameCmd;

		parentFilename << outFilePrefix << "-parents";
		parentFilenameTemp << parentFilename.str() << ".tmp";

		FILE* file;
		file = fopen(parentFilenameTemp.str().c_str(), "w");		
		if (!file) 
		{
			cout << "could not open parent order file" << endl;
			exit(33);
		}

        bool parentFound = false;
		for(int s=0;s< group.size();s++)
		{
			shared_ptr<NEAT::GeneticIndividual> individual = group[s];
			individual->setOrder(s); //tells each org what order in the vector they were in when the were evaluated (tees up the info for the next time)
			int parent1Order = individual->getParent1Order();
			int parent2Order = individual->getParent2Order();
			
			PRINT(parent1Order);
			PRINT(parent2Order);
			
			//write info 
			if(parent1Order > -1)  fprintf(file, "%i ", parent1Order);
			if(parent2Order > -1)  fprintf(file, "%i ", parent2Order);
		    fprintf(file, "\n");				
		
		    if(parent1Order > -1 || parent2Order > -1) parentFound = true;
		    PRINT(parentFound);
						
		}
		fclose(file);
		//mv the temp file to the non-temp name
		if(parentFound) //only print the file if we found a parent (i.e. it is not the first gen)
        {
				cout << "CREATING PARENTS FILE" << endl;
				parentFilenameCmd << "mv " << parentFilenameTemp.str() << " " << parentFilename.str();
				int result = ::system(parentFilenameCmd.str().c_str());
				(void) result;
			
        }
		
	}

    int BracketExperiment::writeVoxelyzeFile( CArray3Df ContinuousArray, CArray3Df PassiveStiffArray, CArray3Df PassiveSoftArray,CArray3Df Phase1Array , CArray3Df Phase2Array, string individualID)
	{
		ofstream md5file;
		md5file.open ("md5sumTMP.txt");		

		std::ostringstream stiffness;
		stiffness << (int(NEAT::Globals::getSingleton()->getParameterValue("MaxStiffness")));
		
		//add in some stuff to allow the addition of obstacles (and an area around the creature for these obstacles)
		int DoObs = (int(NEAT::Globals::getSingleton()->getParameterValue("HasObstacles")));
		int ObsBuf;
		if (DoObs > 0) ObsBuf = 50; else ObsBuf = 0;	//we'll add 50 voxels around the object to allow space (should really make this value a parameter) 

  		ofstream myfile;
		std::ostringstream myFileName;
		myFileName << individualID << "_genome.vxa";
  		myfile.open (myFileName.str().c_str());
//1000000000
  		myfile << "\
<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n\
<VXA Version=\"1.0\">\n\
<Simulator>\n\
<Integration>\n\
<Integrator>0</Integrator>\n\
<DtFrac>0.9</DtFrac>\n\
</Integration>\n\
<Damping>\n\
<BondDampingZ>1</BondDampingZ>\n\
<ColDampingZ>0.8</ColDampingZ>\n\
<SlowDampingZ>0.001</SlowDampingZ>\n\
</Damping>\n\
<Collisions>\n\
<SelfColEnabled>1</SelfColEnabled>\n\
<ColSystem>3</ColSystem>\n\
<CollisionHorizon>2</CollisionHorizon>\n\
</Collisions>\n\
<Features>\n\
<FluidDampEnabled>0</FluidDampEnabled>\n\
<PoissonKickBackEnabled>0</PoissonKickBackEnabled>\n\
<EnforceLatticeEnabled>0</EnforceLatticeEnabled>\n\
</Features>\n\
<SurfMesh>\n\
<CMesh>\n\
<DrawSmooth>1</DrawSmooth>\n\
<Vertices/>\n\
<Facets/>\n\
<Lines/>\n\
</CMesh>\n\
</SurfMesh>\n\
<SimStop>\n\
<SimStopTime>" << (float(NEAT::Globals::getSingleton()->getParameterValue("SimulationTime"))) << "</SimStopTime>\n\
<SimStopStep>1000000000</SimStopStep>\n\
</SimStop>\n\
<Fitness>\n\
<!--Track the center of mass -->\n\
<TrackCOM>0</TrackCOM>\n\
<!-- Specify the fitness filename that will be output -->\n\
<FitnessFileNm>" << individualID << "_fitness.xml</FitnessFileNm>\n\
</Fitness>\n\
</Simulator>\n\
<Environment>\n\
<Fixed_Regions>\n\
<NumFixed>0</NumFixed>\n\
</Fixed_Regions>\n\
<Forced_Regions>\n\
<NumForced>0</NumForced>\n\
</Forced_Regions>\n\
<Gravity>\n\
<GravEnabled>1</GravEnabled>\n\
<GravAcc>-27.468</GravAcc>\n\
<FloorEnabled>1</FloorEnabled>\n\
</Gravity>\n\
<Thermal>\n\
<TempEnabled>1</TempEnabled>\n\
<TempAmp>39</TempAmp>\n\
<TempBase>25</TempBase>\n\
<VaryTempEnabled>1</VaryTempEnabled>\n\
<TempPeriod>0.025</TempPeriod>\n\
</Thermal>\n\
</Environment>\n\
<VXC Version=\"0.93\">\n\
<Lattice>\n\
<Lattice_Dim>" << VoxelSize << "</Lattice_Dim>\n\
<X_Dim_Adj>1</X_Dim_Adj>\n\
<Y_Dim_Adj>1</Y_Dim_Adj>\n\
<Z_Dim_Adj>1</Z_Dim_Adj>\n\
<X_Line_Offset>0</X_Line_Offset>\n\
<Y_Line_Offset>0</Y_Line_Offset>\n\
<X_Layer_Offset>0</X_Layer_Offset>\n\
<Y_Layer_Offset>0</Y_Layer_Offset>\n\
</Lattice>\n\
<Voxel>\n\
<Vox_Name>BOX</Vox_Name>\n\
<X_Squeeze>1</X_Squeeze>\n\
<Y_Squeeze>1</Y_Squeeze>\n\
<Z_Squeeze>1</Z_Squeeze>\n\
</Voxel>\n\
<Palette>\n\
<Material ID=\"1\">\n\
<MatType>0</MatType>\n\
<Name>Active_+</Name>\n\
<Display>\n\
<Red>1</Red>\n\
<Green>0</Green>\n\
<Blue>0</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>1e+007</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>1e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0.01</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
<Material ID=\"2\">\n\
<MatType>0</MatType>\n\
<Name>Passive_Soft</Name>\n\
<Display>\n\
<Red>0</Red>\n\
<Green>1</Green>\n\
<Blue>1</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>1e+007</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>1e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
<Material ID=\"3\">\n\
<MatType>0</MatType>\n\
<Name>Active_-</Name>\n\
<Display>\n\
<Red>0</Red>\n\
<Green>1</Green>\n\
<Blue>0</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>1e+007</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>1e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>-0.01</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
<Material ID=\"4\">\n\
<MatType>0</MatType>\n\
<Name>Passive_Hard</Name>\n\
<Display>\n\
<Red>0</Red>\n\
<Green>0</Green>\n\
<Blue>1</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>" << stiffness.str().c_str() << "e+006</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>1e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
<Material ID=\"5\">\n\
<MatType>0</MatType>\n\
<Name>Obstacle</Name>\n\
<Display>\n\
<Red>1</Red>\n\
<Green>0.47</Green>\n\
<Blue>1</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>1e+007</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Exclude_COM>1</Exclude_COM>\n\
<Density>5e+07</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0.0</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>1</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
</Palette>\n\
<Structure Compression=\"ASCII_READABLE\">\n\
<X_Voxels>" << num_x_voxels+(2*ObsBuf) << "</X_Voxels>\n\
<Y_Voxels>" << num_y_voxels+(2*ObsBuf) << "</Y_Voxels>\n\
<Z_Voxels>" << num_z_voxels << "</Z_Voxels>\n\
<Data>\n\
<Layer><![CDATA[";

CArray3Df ArrayForVoxelyze = createArrayForVoxelyze(ContinuousArray, PassiveStiffArray, PassiveSoftArray, Phase1Array, Phase2Array);

/* nac: marching cubes, finish this! /

std::cout << "Performing marching cubes...\n";
CMesh OutputMesh;
CMarchCube::SingleMaterialMultiColor(&OutputMesh, &ContinuousArray, &rColorArray, &gColorArray, &bColorArray, voxelExistsThreshold, VoxelSize*1000);	//jmc: last argument is the threshold above which we consider the voxel to be extant

//format: 	static void MultiMaterial(CMesh* pMeshOut, void* pArrays, bool SumMat, CColor* pColors = NULL, float Thresh = 0.0f, float Scale = 1.0f);
*/

//some stuff for circular obstacles
float centerX, centerY, ObsDistDes, CurDist;
centerX=(num_x_voxels+(2*ObsBuf))/2;
centerY=(num_y_voxels+(2*ObsBuf))/2;
float ObsClearance = 1.5;	//the number of voxels away from the creature we want the obstacle
float ObsThickness = 1; //how many voxels thick is it?
ObsDistDes=sqrt(pow(num_x_voxels/2,2)+pow(num_y_voxels/2,2))+ObsClearance;	

int v=0;
for (int z=0; z<num_z_voxels; z++)
	{
	for (int y=0; y<num_y_voxels+(2*ObsBuf); y++)
		{
		for (int x=0; x<num_x_voxels+(2*ObsBuf); x++)
			{

			// NOTE: this uses material 0 as passive, material 1 as active phase 1, and material 2 as active phase 2
			/*
			if(ContinuousArray[v] >= Phase1Array[v] && ContinuousArray[v] >= Phase2Array[v] && ContinuousArray[v] >= PassiveArray[v]) myfile << 0;
			else if(PassiveArray[v] >= ContinuousArray[v] && PassiveArray[v] >= Phase2Array[v] && PassiveArray[v] >= Phase1Array[v]) myfile << 1;
			else if(Phase1Array[v] >= ContinuousArray[v] && Phase1Array[v] >= Phase2Array[v] && Phase1Array[v] >= PassiveArray[v]) myfile << 2;
			else if(Phase2Array[v] >= ContinuousArray[v] && Phase2Array[v] >= Phase1Array[v] && Phase2Array[v] >= PassiveArray[v]) myfile << 3;
			*/
			// change "=" to random assignment? currently biases passive, then phase 1, then phase 2 if all equal

			CurDist=sqrt(pow(x-centerX,2)+pow(y-centerY,2));
			
			//if we're within the area occupied by the creature, format the voxels appropriately, else add in zeros
			if ((y>=ObsBuf) && (y<ObsBuf+num_y_voxels) && (x>=ObsBuf) && (x<ObsBuf+num_x_voxels)){
				myfile << ArrayForVoxelyze[v];
				md5file << ArrayForVoxelyze[v];
				v++;
				}
			//if this is a place where we want an obstacle, place it
			else if ((DoObs > 0) && (z==0) && (
				((CurDist>ObsDistDes) && (CurDist<ObsDistDes+ObsThickness))	/*a first concentric circle*/
				|| ((CurDist>ObsDistDes*1.5) && (CurDist<ObsDistDes*1.5+ObsThickness))	/*a second concentric circle*/
				|| ((CurDist>ObsDistDes*2) && (CurDist<ObsDistDes*2+ObsThickness))	/*a third concentric circle*/
				)){
				myfile << 5;
				md5file << 5;
				}
			//o.w., no voxel is placed 
			else {
				myfile << 0;
				md5file << 0;
				}

			}
		}
	if (z<num_z_voxels-1) {myfile << "]]></Layer>\n<Layer><![CDATA["; md5file << "\n";}
	}

myfile << "]]></Layer>\n\
</Data>\n\
</Structure>\n\
</VXC>\n\
</VXA>";

  		myfile.close();
		md5file.close();
		
		// check if "robot" is empty space:
		for (int i=0; i<ArrayForVoxelyze.GetFullSize(); i++)
		{
			if (!ArrayForVoxelyze[i]==0)
			{
				numVoxelsFilled++;
				if (ArrayForVoxelyze[i] == 1 || ArrayForVoxelyze[i] == 3)
				{
					numVoxelsActuated++;
				}
			}
		}
		//if (voxelPenalty > 1) {return true;}
		//else {return false;}
		if (int(NEAT::Globals::getSingleton()->getParameterValue("PenaltyType")) == 3) 
		{
			return numConnections/2;
		} else {
			if (int(NEAT::Globals::getSingleton()->getParameterValue("PenaltyType")) == 2) 
			{
				return numVoxelsActuated;
			} else {
				return numVoxelsFilled;
			}
		}
	}		

	CArray3Df BracketExperiment::createArrayForVoxelyze(CArray3Df ContinuousArray,CArray3Df PassiveStiffArray, CArray3Df PassiveSoftArray,CArray3Df Phase1Array,CArray3Df Phase2Array)
	{	
		CArray3Df ArrayForVoxelyze(num_x_voxels, num_y_voxels, num_z_voxels); //array denoting type of material

		numVoxelsFilled   = 0;
		numVoxelsActuated = 0;
		numConnections    = 0;
		int v=0;
		for (int z=0; z<num_z_voxels; z++)
		{
			for (int y=0; y<num_y_voxels; y++)
			{
				for (int x=0; x<num_x_voxels; x++)
				{
					// NOTE: this uses mateiral 0 as passive, material 1 as active phase 1, and material 2 as active phase 2
		
					if(ContinuousArray[v] < 0)
					{ ArrayForVoxelyze[v] = 0;}
					else if(numMaterials > 1 && PassiveSoftArray[v]>=Phase2Array[v] && PassiveSoftArray[v]>=Phase1Array[v] && PassiveSoftArray[v]>=PassiveStiffArray[v]) 
					{ ArrayForVoxelyze[v] = 2;}
					else if(numMaterials > 3 && PassiveStiffArray[v]>=Phase2Array[v] && PassiveStiffArray[v]>=Phase1Array[v] && PassiveStiffArray[v]>=PassiveSoftArray[v]) 
					{ ArrayForVoxelyze[v] = 4;}
					else if(numMaterials > 0 && Phase1Array[v]>=Phase2Array[v] && Phase1Array[v]>=PassiveSoftArray[v] && Phase1Array[v]>=PassiveStiffArray[v]) 
					{ ArrayForVoxelyze[v] = 1;}
					else if(numMaterials > 2 && Phase2Array[v]>=Phase1Array[v] && Phase2Array[v]>=PassiveSoftArray[v] && Phase2Array[v]>=PassiveStiffArray[v]) 
					{ ArrayForVoxelyze[v] = 3;}
					else 
					{ 
						if (numMaterials > 1) {ArrayForVoxelyze[v] = 2;}
						else {ArrayForVoxelyze[v] = 1;}
					} //if a tie, just use soft passive
		 
					// change "=" to random assignment? currently biases passive, then phase 1, then phase 2 if all equal

					v++;
				}
			}
		}

		ArrayForVoxelyze = makeOneShapeOnly(ArrayForVoxelyze);

		return ArrayForVoxelyze;
	}

	CArray3Df BracketExperiment::makeOneShapeOnly(CArray3Df ArrayForVoxelyze)
	{
			//find start value:
			queue<int> queueToCheck;
			list<int> alreadyChecked;
			pair<  queue<int>, list<int> > queueAndList;
			int startVoxel = int(ArrayForVoxelyze.GetFullSize()/2);
			queueToCheck.push(startVoxel);
			alreadyChecked.push_back(startVoxel);
			while (ArrayForVoxelyze[startVoxel] == 0 && !queueToCheck.empty())
				{
					startVoxel = queueToCheck.front();
					queueAndList = circleOnce(ArrayForVoxelyze, queueToCheck, alreadyChecked);
					queueToCheck = queueAndList.first;
					alreadyChecked = queueAndList.second;
				}

			// create only one shape:	
			alreadyChecked.clear();
			alreadyChecked.push_back(startVoxel);
			//queueToCheck.clear();
			while (!queueToCheck.empty()) {queueToCheck.pop();}
			queueToCheck.push(startVoxel);
			while (!queueToCheck.empty())		
			{
				startVoxel = queueToCheck.front();

				queueAndList = circleOnce(ArrayForVoxelyze, queueToCheck, alreadyChecked);

				queueToCheck = queueAndList.first;
				alreadyChecked = queueAndList.second;

			}


			for (int v=0; v<ArrayForVoxelyze.GetFullSize(); v++)
			{
				if (find(alreadyChecked.begin(),alreadyChecked.end(),v)==alreadyChecked.end())
				{
					ArrayForVoxelyze[v]=0;
				}
			}

		return ArrayForVoxelyze;
	}

	pair< queue<int>, list<int> > BracketExperiment::circleOnce(CArray3Df ArrayForVoxelyze, queue<int> queueToCheck, list<int> alreadyChecked)
	{
				int currentPosition = queueToCheck.front();
				queueToCheck.pop();

				int index;

				index = leftExists(ArrayForVoxelyze, currentPosition);

				if (index>0 && find(alreadyChecked.begin(),alreadyChecked.end(),index)==alreadyChecked.end())
				{queueToCheck.push(index);}
				alreadyChecked.push_back(index);
				index = rightExists(ArrayForVoxelyze, currentPosition);

				if (index>0 && find(alreadyChecked.begin(),alreadyChecked.end(),index)==alreadyChecked.end())
				{queueToCheck.push(index);}
				alreadyChecked.push_back(index);
				index = forwardExists(ArrayForVoxelyze, currentPosition);

				if (index>0 && find(alreadyChecked.begin(),alreadyChecked.end(),index)==alreadyChecked.end())
				{queueToCheck.push(index);}
				alreadyChecked.push_back(index);
				index = backExists(ArrayForVoxelyze, currentPosition);

				if (index>0 && find(alreadyChecked.begin(),alreadyChecked.end(),index)==alreadyChecked.end())
				{queueToCheck.push(index);}
				alreadyChecked.push_back(index);
				index = upExists(ArrayForVoxelyze, currentPosition);

				if (index>0 && find(alreadyChecked.begin(),alreadyChecked.end(),index)==alreadyChecked.end())
				{queueToCheck.push(index);}
				alreadyChecked.push_back(index);
				index = downExists(ArrayForVoxelyze, currentPosition);

				if (index>0 && find(alreadyChecked.begin(),alreadyChecked.end(),index)==alreadyChecked.end())
				{queueToCheck.push(index);}
				alreadyChecked.push_back(index);

				return make_pair (queueToCheck, alreadyChecked);
	}

	int BracketExperiment::leftExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
//cout << "ArrayForVoxelyze.GetFullSize(): "<< ArrayForVoxelyze.GetFullSize() << endl;
				if (currentPosition % num_x_voxels == 0) 
				{
					return -999;	
				}
				else if (ArrayForVoxelyze[currentPosition - 1] == 0)
				{
					 return -999;
				}

				else
				{
				return (currentPosition - 1);
				}
	}

	int BracketExperiment::rightExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (currentPosition % num_x_voxels == num_x_voxels-1) return -999;				
				else if (ArrayForVoxelyze[currentPosition + 1] == 0) return -999;
				else {numConnections++; return (currentPosition + 1);}
	}
			
	int BracketExperiment::forwardExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (int(currentPosition/num_x_voxels) % num_y_voxels == num_y_voxels-1) return -999;				
				else if (ArrayForVoxelyze[currentPosition + num_x_voxels] == 0) return -999;
				else {numConnections++; return (currentPosition + num_x_voxels);}
	}

	int BracketExperiment::backExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (int(currentPosition/num_x_voxels) % num_y_voxels == 0) return -999;				
				else if (ArrayForVoxelyze[currentPosition - num_x_voxels] == 0) return -999;
				else {numConnections++; return (currentPosition - num_x_voxels);}
	}

	int BracketExperiment::upExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (int(currentPosition/(num_x_voxels*num_y_voxels)) % num_z_voxels == num_z_voxels-1) return -999;				
				else if (ArrayForVoxelyze[currentPosition + num_x_voxels*num_y_voxels] == 0) return -999;
				else {numConnections++; return (currentPosition + num_x_voxels*num_y_voxels);}
	}

	int BracketExperiment::downExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (int(currentPosition/(num_x_voxels*num_y_voxels)) % num_z_voxels == 0) return -999;				
				else if (ArrayForVoxelyze[currentPosition - num_x_voxels*num_y_voxels] == 0) return -999;
				else {numConnections++; return (currentPosition - num_x_voxels*num_y_voxels);}
	}


}
