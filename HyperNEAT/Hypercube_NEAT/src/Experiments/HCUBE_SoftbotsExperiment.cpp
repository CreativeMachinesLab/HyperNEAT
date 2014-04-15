#include "MarchCube.h"
#include <iomanip> 
#include "Experiments/HCUBE_SoftbotsExperiment.h"
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
#include <algorithm> 
#include <sys/wait.h>


#define SHAPES_EXPERIMENT_DEBUG (0)
#define PI 3.14159265

namespace HCUBE
{
    using namespace NEAT;

    SoftbotsExperiment::SoftbotsExperiment(string _experimentName)
    :   Experiment(_experimentName)
    {
        
        cout << "Constructing experiment named SoftbotsExperiment.\n";

		addDistanceFromCenter = true;
		addDistanceFromCenterXY = true;
		addDistanceFromCenterYZ = true;
		addDistanceFromCenterXZ = true;

		addAngularPositionXY = true;
		addAngularPositionYZ = true;
		addAngularPositionXZ = true;

		evolvingNeuralNet = false;  // make command line switch
		evolvingSensorsAndMotors = false;
		phaseDrivenActuation = true;

		numPacemaker = int(NEAT::Globals::getSingleton()->getParameterValue("NumPacemaker"));
		cout << "*** NUMPACEMAKER:" << numPacemaker << endl;
			
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

    NEAT::GeneticPopulation* SoftbotsExperiment::createInitialPopulation(int populationSize)
    {

        NEAT::GeneticPopulation* population = new NEAT::GeneticPopulation();

        vector<GeneticNodeGene> genes;
	// cout << "JMC HERE" << endl;

        genes.push_back(GeneticNodeGene("Bias","NetworkSensor",0,false));
        genes.push_back(GeneticNodeGene("x","NetworkSensor",0,false));
        genes.push_back(GeneticNodeGene("y","NetworkSensor",0,false));
        genes.push_back(GeneticNodeGene("z","NetworkSensor",0,false));
        if(evolvingNeuralNet)  genes.push_back(GeneticNodeGene("synapseStartX","NetworkSensor",0,false));
        if(evolvingNeuralNet)  genes.push_back(GeneticNodeGene("synapseStartY","NetworkSensor",0,false));
        if(evolvingNeuralNet)  genes.push_back(GeneticNodeGene("synapseStartZ","NetworkSensor",0,false));
        if(evolvingNeuralNet)  genes.push_back(GeneticNodeGene("synapseStartL","NetworkSensor",0,false));
        if(evolvingNeuralNet)  genes.push_back(GeneticNodeGene("synapseEndX","NetworkSensor",0,false));
        if(evolvingNeuralNet)  genes.push_back(GeneticNodeGene("synapseEndY","NetworkSensor",0,false));
        if(evolvingNeuralNet)  genes.push_back(GeneticNodeGene("synapseEndZ","NetworkSensor",0,false));
        if(evolvingNeuralNet)  genes.push_back(GeneticNodeGene("synapseEndL","NetworkSensor",0,false));
		if(addDistanceFromCenter)	genes.push_back(GeneticNodeGene("d","NetworkSensor",0,false));
		if(addDistanceFromCenterXY) genes.push_back(GeneticNodeGene("dxy","NetworkSensor",0,false));
		if(addDistanceFromCenterYZ) genes.push_back(GeneticNodeGene("dyz","NetworkSensor",0,false));
		if(addDistanceFromCenterXZ) genes.push_back(GeneticNodeGene("dxz","NetworkSensor",0,false));
		if(addAngularPositionXY) genes.push_back(GeneticNodeGene("rxy","NetworkSensor",0,false));
		if(addAngularPositionYZ) genes.push_back(GeneticNodeGene("ryz","NetworkSensor",0,false));
		if(addAngularPositionXZ) genes.push_back(GeneticNodeGene("rxz","NetworkSensor",0,false));
		genes.push_back(GeneticNodeGene("OutputEmpty","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (numMaterials > 3 ) genes.push_back(GeneticNodeGene("OutputPassiveStiff","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (numMaterials > 1 ) genes.push_back(GeneticNodeGene("OutputPassiveSoft","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (numMaterials > 0 ) genes.push_back(GeneticNodeGene("OutputActivePhase1","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (numMaterials > 2 ) genes.push_back(GeneticNodeGene("OutputActivePhase2","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (evolvingSensorsAndMotors ) genes.push_back(GeneticNodeGene("OutputSensor","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (evolvingSensorsAndMotors ) genes.push_back(GeneticNodeGene("OutputMotor","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (evolvingNeuralNet) genes.push_back(GeneticNodeGene("thisSynapseWeight","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (evolvingNeuralNet) genes.push_back(GeneticNodeGene("thisSynapsePresence","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (evolvingNeuralNet) genes.push_back(GeneticNodeGene("nodeTimeConstant","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
		if (evolvingNeuralNet) genes.push_back(GeneticNodeGene("nodeBias","NetworkOutputNode",1,false,ACTIVATION_FUNCTION_SIGMOID));
        for (size_t a=0;a<populationSize;a++)
        {
            shared_ptr<GeneticIndividual> individual(new GeneticIndividual(genes,true,1.0));

            //for (size_t b=0;b<0;b++)  // this was uncommented in jason's original code, but I think since he has b<0 it never fired. setting it to >0 would probably testMutate it
            //{            
            //    individual->testMutate();
            //}
            // std::ostringstream md5initString;
            // md5initString << "md5init" << Globals::getSingleton()->getRandom().getRandomInt(9999999999);

            // individual->setThismd5(md5initString.str());

            population->addIndividual(individual);
        }

        return population;
    }


    double SoftbotsExperiment::mapXYvalToNormalizedGridCoord(const int & r_xyVal, const int & r_numVoxelsXorY) 
    {
        // turn the xth or yth node into its coordinates on a grid from -1 to 1, e.g. x values (1,2,3,4,5) become (-1, -.5 , 0, .5, 1)
        // this works with even numbers, and for x or y grids only 1 wide/tall, which was not the case for the original
        // e.g. see findCluster for the orignal versions where it was not a funciton and did not work with odd or 1 tall/wide #s
		
        double coord;
                
        if(r_numVoxelsXorY==1) coord = 0;
        else                  coord = -1 + ( r_xyVal * 2.0/(r_numVoxelsXorY-1) );

        return(coord);    
    }
	

    double SoftbotsExperiment::processEvaluation(shared_ptr<NEAT::GeneticIndividual> individual, string individualID, int genNum, bool saveVxaOnly, double bestFit)
    {
		cout << "Processing Evaluation" << endl;
		
		//initializes continuous space array with zeros. +1 is because we need to sample
		// these points at all corners of each voxel, leading to n+1 points in any dimension
		CArray3Df ContinuousArray(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df PassiveStiffArray(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df PassiveSoftArray(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df Phase1Array(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df Phase2Array(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df PhaseSensorArray(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array
		CArray3Df PhaseMotorArray(num_x_voxels, num_y_voxels, num_z_voxels); //evolved array


		NEAT::FastNetwork <double> network = individual->spawnFastPhenotypeStack<double>();        //JMC: this is the CPPN network

		int px, py, pz; //temporary variable to store locations as we iterate
		float xNormalized;
		float yNormalized;
		float zNormalized;
		float distanceFromCenter;
		float distanceFromCenterXY;
		float distanceFromCenterYZ;
		float distanceFromCenterXZ;
		float angularPositionXY;
		float angularPositionYZ;
		float angularPositionXZ;
		float thisSynapseWeight;
		float thisSynapsePresence;
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
			if(addAngularPositionXY) { if (yNormalized==0.0 and xNormalized==0.0) { angularPositionXY = 0.0; }	else { angularPositionXY = atan2(double(yNormalized),double(xNormalized))/PI; } }
			if(addAngularPositionYZ) { if (yNormalized==0.0 and zNormalized==0.0) { angularPositionYZ = 0.0; }	else { angularPositionYZ = atan2(double(zNormalized),double(yNormalized))/PI; } }
			if(addAngularPositionXZ) { if (zNormalized==0.0 and xNormalized==0.0) { angularPositionXZ = 0.0; }	else { angularPositionXZ = atan2(double(zNormalized),double(xNormalized))/PI; } }			
	
			network.reinitialize();								//reset CPPN
			network.setValue("x",xNormalized);					//set the input numbers
			network.setValue("y",yNormalized);
			network.setValue("z",zNormalized);
			if(addDistanceFromCenter) network.setValue("d",distanceFromCenter);
			if(addDistanceFromCenterXY) network.setValue("dxy",distanceFromCenterXY);
			if(addDistanceFromCenterYZ) network.setValue("dyz",distanceFromCenterYZ);
			if(addDistanceFromCenterXZ) network.setValue("dxz",distanceFromCenterXZ);
			if(addAngularPositionXY) network.setValue("rxy",angularPositionXY);
			if(addAngularPositionYZ) network.setValue("ryz",angularPositionYZ);
			if(addAngularPositionXZ) network.setValue("rxz",angularPositionXZ);
			if(evolvingNeuralNet) network.setValue("synapseStartX",0);
			if(evolvingNeuralNet) network.setValue("synapseStartY",0);
			if(evolvingNeuralNet) network.setValue("synapseStartZ",0);
			if(evolvingNeuralNet) network.setValue("synapseStartL",0);
			if(evolvingNeuralNet) network.setValue("synapseEndX",0);
			if(evolvingNeuralNet) network.setValue("synapseEndY",0);
			if(evolvingNeuralNet) network.setValue("synapseEndZ",0);
			if(evolvingNeuralNet) network.setValue("synapseEndL",0);
			//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
			//if(addInsideOrOutside) network.setValue("inout",e);
			network.setValue("Bias",0.3);                       
							
			network.update();                                   //JMC: on this line we run the CPPN network...  
			
			ContinuousArray[j] = network.getValue("OutputEmpty");        //JMC: and here we get the CPPN output (which is the weight of the connection between the two)
			if (numMaterials > 3 ) PassiveStiffArray[j] = network.getValue("OutputPassiveStiff");
			if (numMaterials > 1 ) PassiveSoftArray[j] = network.getValue("OutputPassiveSoft");
			if (numMaterials > 0 ) Phase1Array[j] = network.getValue("OutputActivePhase1");
			if (numMaterials > 2 ) Phase2Array[j] = network.getValue("OutputActivePhase2");
			if (evolvingSensorsAndMotors ) PhaseSensorArray[j] = network.getValue("OutputSensor");
			if (evolvingSensorsAndMotors ) PhaseMotorArray[j] = network.getValue("OutputMotor");

		}

		// cout<<"here1"<<endl;

		CArray3Df ArrayForVoxelyze = createArrayForVoxelyze(ContinuousArray, PassiveStiffArray, PassiveSoftArray, Phase1Array, Phase2Array, PhaseSensorArray, PhaseMotorArray);
		CArray3Df SensorArrayForVoxelyze = createSensorArrayForVoxelyze(PhaseSensorArray, ArrayForVoxelyze);
		CArray3Df MuscleArrayForVoxelyze = createSensorArrayForVoxelyze(PhaseMotorArray, ArrayForVoxelyze);

		
		// float synapseWeights [numSensors*numHidden+numHidden*numHidden+numHidden*numMuscles];
		float synapseWeights [100000];
		MuscleArrayForVoxelyze = numberSensors(MuscleArrayForVoxelyze);
	    SensorArrayForVoxelyze = numberSensors(SensorArrayForVoxelyze);
	    int numSensors = 0;
	    int numHidden = 0;
	    int numMuscles = 0;
	    float sensorPositions [100][3];
	    float hiddenPositions [100][3];
	    float musclePositions [100][3];
	    float nodeTimeConstant [300];
	    float nodeBias [300];
		
		if (evolvingSensorsAndMotors)
		{

			int v;		
	        // cout<<"original array:"<<endl;
	        // // print this:
	        // // int v=0;
	        // v=0;
	        // for (int xIndex=0; xIndex<num_x_voxels; xIndex++)
	        // {
	        //     for (int yIndex=0; yIndex<num_y_voxels; yIndex++)
	        //     {
	        //         for (int zIndex=0; zIndex<num_z_voxels; zIndex++)
	        //         {
	        //             cout << ArrayForVoxelyze[v] << " ";
	        //             v++;
	        //         }
	        //         cout << endl;
	        // 	}
	        //     cout << endl << endl;
	        // }
	        	       
	        // v=0;
	        // for (int xIndex=0; xIndex<num_x_voxels; xIndex++)
	        // {
	        //     for (int yIndex=0; yIndex<num_y_voxels; yIndex++)
	        //     {
	        //         for (int zIndex=0; zIndex<num_z_voxels; zIndex++)
	        //         {
	        //             if (SensorArrayForVoxelyze[v] > numSensors) {numSensors = SensorArrayForVoxelyze[v];}
	        //             if (MuscleArrayForVoxelyze[v] > numMuscles) {numMuscles = MuscleArrayForVoxelyze[v];}
	        //             v++;
	        //         }
	        //     }
	        // }
	        for (int j=0; j<ContinuousArray.GetFullSize(); j++)  //iterate through each location of the continuous matrix
			{ 
				if (SensorArrayForVoxelyze[j] > numSensors) {numSensors = SensorArrayForVoxelyze[j];}
	            if (MuscleArrayForVoxelyze[j] > numMuscles) {numMuscles = MuscleArrayForVoxelyze[j];}
	        }


			// cout<<"sensors:"<<endl;
			// // print this:
			// // int v=0;
			// v=0;
			// for (int xIndex=0; xIndex<num_x_voxels; xIndex++)
			// {
			// 	for (int yIndex=0; yIndex<num_y_voxels; yIndex++)
			// 	{
			// 		for (int zIndex=0; zIndex<num_z_voxels; zIndex++)
			// 		{
			// 			cout << SensorArrayForVoxelyze[v] << " ";
			// 			v++;
			// 		}
			// 		cout << endl;
			// 	}
			// 	cout << endl;
			// }

	        // cout<<"numSensors: "<< numSensors <<endl;

	       	// numSensors += numPacemaker;

	       	// cout<<"numSensors,includingPacemaker: "<< numSensors <<endl;

	        // cout<<"numMuscles: "<< numMuscles <<endl;

	        if (numSensors == 0) {cout << "No Sensors... assigning fitness of 0.00001" << endl; return 0.00001;}  // exit if no muscles or sensors
	        if (numMuscles == 0) {cout << "No Muscles... assigning fitness of 0.00001" << endl; return 0.00001;}  // exit if no muscles or sensors
	        if (numSensors > 99) {cout << "Over 100 Sensors... assigning fitness of 0.00001" << endl; return 0.00001;}  // exit if no muscles or sensors
	        if (numMuscles > 99) {cout << "Over 100 Muscles... assigning fitness of 0.00001" << endl; return 0.00001;}  // exit if no muscles or sensors

	        float thisX,thisY,thisZ;
	        int numVoxelsInSensor;
	        // float sensorPositions [numSensors][3];
	        for (int thisSensor=1; thisSensor<numSensors+1; thisSensor++)
	        {
	        	numVoxelsInSensor = 0;
				thisX = 0.0;
				thisY = 0.0;
				thisZ = 0.0;
		        for (int j=0; j<ContinuousArray.GetFullSize(); j++)  //iterate through each location of the continuous matrix
				{
					if (SensorArrayForVoxelyze[j] == thisSensor)
					{
						ContinuousArray.GetXYZ(&px, &py, &pz, j); //gets XYZ location of this element in the array
						thisX += mapXYvalToNormalizedGridCoord(px, num_x_voxels);
						thisY += mapXYvalToNormalizedGridCoord(py, num_y_voxels);
						thisZ += mapXYvalToNormalizedGridCoord(pz, num_z_voxels);
						numVoxelsInSensor++;
					}
				}
				// cout << "avg for sensor " << thisSensor <<": "<< thisX/numVoxelsInSensor << ", " << thisY/numVoxelsInSensor << ", " << thisZ/numVoxelsInSensor << endl;
				if (numVoxelsInSensor == 0)
				{
					sensorPositions[thisSensor-1][0] = 0;
					sensorPositions[thisSensor-1][1] = 0;
					sensorPositions[thisSensor-1][2] = 0;
				}
				else
				{
					sensorPositions[thisSensor-1][0] = thisX/numVoxelsInSensor;
					sensorPositions[thisSensor-1][1] = thisY/numVoxelsInSensor;
					sensorPositions[thisSensor-1][2] = thisZ/numVoxelsInSensor;
				}
	        }

	        int numVoxelsInMuscle;
	        // float musclePositions [numMuscles][3];
	        for (int thisMuscle=1; thisMuscle<numMuscles+1; thisMuscle++)
	        {
	        	numVoxelsInMuscle = 0;
				thisX = 0;
				thisY = 0;
				thisZ = 0;
		        for (int j=0; j<ContinuousArray.GetFullSize(); j++)  //iterate through each location of the continuous matrix
				{
					if (MuscleArrayForVoxelyze[j] == thisMuscle)
					{
						ContinuousArray.GetXYZ(&px, &py, &pz, j); //gets XYZ location of this element in the array
						thisX += mapXYvalToNormalizedGridCoord(px, num_x_voxels);
						thisY += mapXYvalToNormalizedGridCoord(py, num_y_voxels);
						thisZ += mapXYvalToNormalizedGridCoord(pz, num_z_voxels);
						numVoxelsInMuscle++;
					}
				}
				// cout << "avg for Muscle " << thisMuscle <<": "<< thisX/numVoxelsInMuscle << ", " << thisY/numVoxelsInMuscle << ", " << thisZ/numVoxelsInMuscle << endl;
				musclePositions[thisMuscle-1][0] = thisX/numVoxelsInMuscle;
				musclePositions[thisMuscle-1][1] = thisY/numVoxelsInMuscle;
				musclePositions[thisMuscle-1][2] = thisZ/numVoxelsInMuscle;
	        }

	        numHidden = round(float(numSensors+numMuscles)/2);
	        // float hiddenPositions [numHidden][3];
	        // cout << "numHidden: " << numHidden << endl;
	        int spacing = 0;
	        while (pow(spacing,3) < numHidden) {spacing++;}
	        // cout << "spacing: " << spacing << endl;
	        float posX = 2.0/(spacing+1) - 1;
	        float posY = 2.0/(spacing+1) - 1;
	        float posZ = 2.0/(spacing+1) - 1;
	        for (int thisHidden=0; thisHidden<numHidden; thisHidden++)
	        {
	        	hiddenPositions[thisHidden][0] = posX;
	        	hiddenPositions[thisHidden][1] = posY;
	        	hiddenPositions[thisHidden][2] = posZ;
	        	posX += 2.0/(spacing+1);
	        	if (posX>=.99)
	        	{
	        		posX = 2.0/(spacing+1) - 1;
	        		posY += 2.0/(spacing+1);
	        	}
	        	if (posY>=.99)
	        	{
	        		posY = 2.0/(spacing+1) - 1;
	        		posZ += 2.0/(spacing+1);
	        	}
	        }

	        // for (int i=0; i<numSensors; i++)
	        // {
	        // 	cout << "sensor " << i << " location: " << sensorPositions[i][0] << ", " << sensorPositions[i][1] << ", " << sensorPositions[i][2] << endl;
	        // }
	        // for (int i=0; i<numHidden; i++)
	        // {
	        // 	cout << "hidden " << i << " location: " << hiddenPositions[i][0] << ", " << hiddenPositions[i][1] << ", " << hiddenPositions[i][2] << endl;
	        // }
	        // for (int i=0; i<numMuscles; i++)
	        // {
	        // 	cout << "muscle " << i << " location: " << musclePositions[i][0] << ", " << musclePositions[i][1] << ", " << musclePositions[i][2] << endl;
	        // }

	        // float synapseWeights [numSensors*numHidden+numHidden*numHidden+numHidden*numMuscles];


			for (int startNeuron=0; startNeuron<numSensors; startNeuron++)  
			{ 

				network.reinitialize();								//reset CPPN
				network.setValue("x",0);					//set the input numbers
				network.setValue("y",0);
				network.setValue("z",0);
				if(addDistanceFromCenter) network.setValue("d",0);
				if(addDistanceFromCenterXY) network.setValue("dxy",0);
				if(addDistanceFromCenterYZ) network.setValue("dyz",0);
				if(addDistanceFromCenterXZ) network.setValue("dxz",0);
				if(addAngularPositionXY) network.setValue("rxy",0);
				if(addAngularPositionYZ) network.setValue("ryz",0);
				if(addAngularPositionXZ) network.setValue("rxz",0);
				if(evolvingNeuralNet) network.setValue("synapseStartX",sensorPositions[startNeuron][0]);
				if(evolvingNeuralNet) network.setValue("synapseStartY",sensorPositions[startNeuron][1]);
				if(evolvingNeuralNet) network.setValue("synapseStartZ",sensorPositions[startNeuron][2]);
				if(evolvingNeuralNet) network.setValue("synapseStartL",-1.0);
				if(evolvingNeuralNet) network.setValue("synapseEndX",0);
				if(evolvingNeuralNet) network.setValue("synapseEndY",0);
				if(evolvingNeuralNet) network.setValue("synapseEndZ",0);
				if(evolvingNeuralNet) network.setValue("synapseEndL",0);
				//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
				//if(addInsideOrOutside) network.setValue("inout",e);

				network.setValue("Bias",0.3); 

				network.update();                                   //JMC: on this line we run the CPPN network...  

				nodeBias[startNeuron] = network.getValue("nodeBias");

				network.reinitialize();								//reset CPPN
				network.setValue("x",0);					//set the input numbers
				network.setValue("y",0);
				network.setValue("z",0);
				if(addDistanceFromCenter) network.setValue("d",0);
				if(addDistanceFromCenterXY) network.setValue("dxy",0);
				if(addDistanceFromCenterYZ) network.setValue("dyz",0);
				if(addDistanceFromCenterXZ) network.setValue("dxz",0);
				if(addAngularPositionXY) network.setValue("rxy",0);
				if(addAngularPositionYZ) network.setValue("ryz",0);
				if(addAngularPositionXZ) network.setValue("rxz",0);
				if(evolvingNeuralNet) network.setValue("synapseStartX",0);
				if(evolvingNeuralNet) network.setValue("synapseStartY",0);
				if(evolvingNeuralNet) network.setValue("synapseStartZ",0);
				if(evolvingNeuralNet) network.setValue("synapseStartL",0);
				if(evolvingNeuralNet) network.setValue("synapseEndX",sensorPositions[startNeuron][0]);
				if(evolvingNeuralNet) network.setValue("synapseEndY",sensorPositions[startNeuron][1]);
				if(evolvingNeuralNet) network.setValue("synapseEndZ",sensorPositions[startNeuron][2]);
				if(evolvingNeuralNet) network.setValue("synapseEndL",-1.0);
				//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
				//if(addInsideOrOutside) network.setValue("inout",e);
				network.setValue("Bias",0.3);                       
								
				network.update();                                   //JMC: on this line we run the CPPN network...  
				
				nodeTimeConstant[startNeuron] = pow(10,network.getValue("nodeTimeConstant"));

			}

			for (int startNeuron=numSensors; startNeuron<numSensors+numHidden; startNeuron++)  
			{ 
				network.reinitialize();								//reset CPPN
				network.setValue("x",0);					//set the input numbers
				network.setValue("y",0);
				network.setValue("z",0);
				if(addDistanceFromCenter) network.setValue("d",0);
				if(addDistanceFromCenterXY) network.setValue("dxy",0);
				if(addDistanceFromCenterYZ) network.setValue("dyz",0);
				if(addDistanceFromCenterXZ) network.setValue("dxz",0);
				if(addAngularPositionXY) network.setValue("rxy",0);
				if(addAngularPositionYZ) network.setValue("ryz",0);
				if(addAngularPositionXZ) network.setValue("rxz",0);
				if(evolvingNeuralNet) network.setValue("synapseStartX",hiddenPositions[startNeuron][0]);
				if(evolvingNeuralNet) network.setValue("synapseStartY",hiddenPositions[startNeuron][1]);
				if(evolvingNeuralNet) network.setValue("synapseStartZ",hiddenPositions[startNeuron][2]);
				if(evolvingNeuralNet) network.setValue("synapseStartL",0);
				if(evolvingNeuralNet) network.setValue("synapseEndX",0);
				if(evolvingNeuralNet) network.setValue("synapseEndY",0);
				if(evolvingNeuralNet) network.setValue("synapseEndZ",0);
				if(evolvingNeuralNet) network.setValue("synapseEndL",0);
				//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
				//if(addInsideOrOutside) network.setValue("inout",e);
				network.setValue("Bias",0.3);                       
								
				network.update();                                   //JMC: on this line we run the CPPN network...  
				
				nodeBias[startNeuron] = network.getValue("nodeBias");

				network.reinitialize();								//reset CPPN
				network.setValue("x",0);					//set the input numbers
				network.setValue("y",0);
				network.setValue("z",0);
				if(addDistanceFromCenter) network.setValue("d",0);
				if(addDistanceFromCenterXY) network.setValue("dxy",0);
				if(addDistanceFromCenterYZ) network.setValue("dyz",0);
				if(addDistanceFromCenterXZ) network.setValue("dxz",0);
				if(addAngularPositionXY) network.setValue("rxy",0);
				if(addAngularPositionYZ) network.setValue("ryz",0);
				if(addAngularPositionXZ) network.setValue("rxz",0);
				if(evolvingNeuralNet) network.setValue("synapseStartX",0);
				if(evolvingNeuralNet) network.setValue("synapseStartY",0);
				if(evolvingNeuralNet) network.setValue("synapseStartZ",0);
				if(evolvingNeuralNet) network.setValue("synapseStartL",0);
				if(evolvingNeuralNet) network.setValue("synapseEndX",hiddenPositions[startNeuron][0]);
				if(evolvingNeuralNet) network.setValue("synapseEndY",hiddenPositions[startNeuron][1]);
				if(evolvingNeuralNet) network.setValue("synapseEndZ",hiddenPositions[startNeuron][2]);
				if(evolvingNeuralNet) network.setValue("synapseEndL",0);
				//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
				//if(addInsideOrOutside) network.setValue("inout",e);
				network.setValue("Bias",0.3);                       
								
				network.update();                                   //JMC: on this line we run the CPPN network...  
				
				nodeTimeConstant[startNeuron] = pow(10,network.getValue("nodeTimeConstant"));			
			}

			for (int startNeuron=numSensors+numHidden; startNeuron<numSensors+numHidden+numMuscles; startNeuron++)  
			{ 
				network.reinitialize();								//reset CPPN
				network.setValue("x",0);					//set the input numbers
				network.setValue("y",0);
				network.setValue("z",0);
				if(addDistanceFromCenter) network.setValue("d",0);
				if(addDistanceFromCenterXY) network.setValue("dxy",0);
				if(addDistanceFromCenterYZ) network.setValue("dyz",0);
				if(addDistanceFromCenterXZ) network.setValue("dxz",0);
				if(addAngularPositionXY) network.setValue("rxy",0);
				if(addAngularPositionYZ) network.setValue("ryz",0);
				if(addAngularPositionXZ) network.setValue("rxz",0);
				if(evolvingNeuralNet) network.setValue("synapseStartX",musclePositions[startNeuron][0]);
				if(evolvingNeuralNet) network.setValue("synapseStartY",musclePositions[startNeuron][1]);
				if(evolvingNeuralNet) network.setValue("synapseStartZ",musclePositions[startNeuron][2]);
				if(evolvingNeuralNet) network.setValue("synapseStartL",1.0);
				if(evolvingNeuralNet) network.setValue("synapseEndX",0);
				if(evolvingNeuralNet) network.setValue("synapseEndY",0);
				if(evolvingNeuralNet) network.setValue("synapseEndZ",0);
				if(evolvingNeuralNet) network.setValue("synapseEndL",0);
				//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
				//if(addInsideOrOutside) network.setValue("inout",e);
				network.setValue("Bias",0.3);                       
								
				network.update();                                   //JMC: on this line we run the CPPN network...  
				
				nodeBias[startNeuron] = network.getValue("nodeBias");

				network.reinitialize();								//reset CPPN
				network.setValue("x",0);					//set the input numbers
				network.setValue("y",0);
				network.setValue("z",0);
				if(addDistanceFromCenter) network.setValue("d",0);
				if(addDistanceFromCenterXY) network.setValue("dxy",0);
				if(addDistanceFromCenterYZ) network.setValue("dyz",0);
				if(addDistanceFromCenterXZ) network.setValue("dxz",0);
				if(addAngularPositionXY) network.setValue("rxy",0);
				if(addAngularPositionYZ) network.setValue("ryz",0);
				if(addAngularPositionXZ) network.setValue("rxz",0);
				if(evolvingNeuralNet) network.setValue("synapseStartX",0);
				if(evolvingNeuralNet) network.setValue("synapseStartY",0);
				if(evolvingNeuralNet) network.setValue("synapseStartZ",0);
				if(evolvingNeuralNet) network.setValue("synapseStartL",0);
				if(evolvingNeuralNet) network.setValue("synapseEndX",musclePositions[startNeuron][0]);
				if(evolvingNeuralNet) network.setValue("synapseEndY",musclePositions[startNeuron][1]);
				if(evolvingNeuralNet) network.setValue("synapseEndZ",musclePositions[startNeuron][2]);
				if(evolvingNeuralNet) network.setValue("synapseEndL",1.0);
				//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
				//if(addInsideOrOutside) network.setValue("inout",e);
				network.setValue("Bias",0.3);                       
								
				network.update();                                   //JMC: on this line we run the CPPN network...  
				
				nodeTimeConstant[startNeuron] = pow(10,network.getValue("nodeTimeConstant"));
			}

	        for (int startNeuron=0; startNeuron<numSensors; startNeuron++)  
			{ 
				for (int endNeuron = 0; endNeuron<numHidden; endNeuron++)
				{
					network.reinitialize();								//reset CPPN
					network.setValue("x",0);					//set the input numbers
					network.setValue("y",0);
					network.setValue("z",0);
					if(addDistanceFromCenter) network.setValue("d",0);
					if(addDistanceFromCenterXY) network.setValue("dxy",0);
					if(addDistanceFromCenterYZ) network.setValue("dyz",0);
					if(addDistanceFromCenterXZ) network.setValue("dxz",0);
					if(addAngularPositionXY) network.setValue("rxy",0);
					if(addAngularPositionYZ) network.setValue("ryz",0);
					if(addAngularPositionXZ) network.setValue("rxz",0);
					if(evolvingNeuralNet) network.setValue("synapseStartX",sensorPositions[startNeuron][0]);
					if(evolvingNeuralNet) network.setValue("synapseStartY",sensorPositions[startNeuron][1]);
					if(evolvingNeuralNet) network.setValue("synapseStartZ",sensorPositions[startNeuron][2]);
					if(evolvingNeuralNet) network.setValue("synapseStartL",-1.0);
					if(evolvingNeuralNet) network.setValue("synapseEndX",hiddenPositions[endNeuron][0]);
					if(evolvingNeuralNet) network.setValue("synapseEndY",hiddenPositions[endNeuron][1]);
					if(evolvingNeuralNet) network.setValue("synapseEndZ",hiddenPositions[endNeuron][2]);
					if(evolvingNeuralNet) network.setValue("synapseEndL",0);
					//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
					//if(addInsideOrOutside) network.setValue("inout",e);
					network.setValue("Bias",0.3);                       
									
					network.update();                                   //JMC: on this line we run the CPPN network...  

					if (true)//network.getValue("thisSynapsePresence")>0)
					{
						synapseWeights[startNeuron*numHidden + endNeuron] = network.getValue("thisSynapseWeight");
					}
					else
					{
						synapseWeights[startNeuron*numHidden + endNeuron] = 0;
					}
				}
			}

			for (int startNeuron=0; startNeuron<numHidden; startNeuron++)  
			{ 
				for (int endNeuron = 0; endNeuron<numHidden; endNeuron++)
				{
					network.reinitialize();								//reset CPPN
					network.setValue("x",0);					//set the input numbers
					network.setValue("y",0);
					network.setValue("z",0);
					if(addDistanceFromCenter) network.setValue("d",0);
					if(addDistanceFromCenterXY) network.setValue("dxy",0);
					if(addDistanceFromCenterYZ) network.setValue("dyz",0);
					if(addDistanceFromCenterXZ) network.setValue("dxz",0);
					if(addAngularPositionXY) network.setValue("rxy",0);
					if(addAngularPositionYZ) network.setValue("ryz",0);
					if(addAngularPositionXZ) network.setValue("rxz",0);
					if(evolvingNeuralNet) network.setValue("synapseStartX",hiddenPositions[startNeuron][0]);
					if(evolvingNeuralNet) network.setValue("synapseStartY",hiddenPositions[startNeuron][1]);
					if(evolvingNeuralNet) network.setValue("synapseStartZ",hiddenPositions[startNeuron][2]);
					if(evolvingNeuralNet) network.setValue("synapseStartL",0);
					if(evolvingNeuralNet) network.setValue("synapseEndX",hiddenPositions[endNeuron][0]);
					if(evolvingNeuralNet) network.setValue("synapseEndY",hiddenPositions[endNeuron][1]);
					if(evolvingNeuralNet) network.setValue("synapseEndZ",hiddenPositions[endNeuron][2]);
					if(evolvingNeuralNet) network.setValue("synapseEndL",0);
					//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
					//if(addInsideOrOutside) network.setValue("inout",e);
					network.setValue("Bias",0.3);                       
									
					network.update();                                   //JMC: on this line we run the CPPN network...  

					if (true)//network.getValue("thisSynapsePresence")>0)
					{
						synapseWeights[numSensors*numHidden + startNeuron*numHidden + endNeuron] = network.getValue("thisSynapseWeight");
					}
					else
					{
						synapseWeights[numSensors*numHidden + startNeuron*numHidden + endNeuron] = 0;
					}
				}
			}

			for (int startNeuron=0; startNeuron<numHidden; startNeuron++)  
			{ 
				for (int endNeuron = 0; endNeuron<numMuscles; endNeuron++)
				{
					network.reinitialize();								//reset CPPN
					network.setValue("x",0);					//set the input numbers
					network.setValue("y",0);
					network.setValue("z",0);
					if(addDistanceFromCenter) network.setValue("d",0);
					if(addDistanceFromCenterXY) network.setValue("dxy",0);
					if(addDistanceFromCenterYZ) network.setValue("dyz",0);
					if(addDistanceFromCenterXZ) network.setValue("dxz",0);
					if(addAngularPositionXY) network.setValue("rxy",0);
					if(addAngularPositionYZ) network.setValue("ryz",0);
					if(addAngularPositionXZ) network.setValue("rxz",0);
					if(evolvingNeuralNet) network.setValue("synapseStartX",hiddenPositions[startNeuron][0]);
					if(evolvingNeuralNet) network.setValue("synapseStartY",hiddenPositions[startNeuron][1]);
					if(evolvingNeuralNet) network.setValue("synapseStartZ",hiddenPositions[startNeuron][2]);
					if(evolvingNeuralNet) network.setValue("synapseStartL",0);
					if(evolvingNeuralNet) network.setValue("synapseEndX",musclePositions[endNeuron-numHidden][0]);
					if(evolvingNeuralNet) network.setValue("synapseEndY",musclePositions[endNeuron-numHidden][1]);
					if(evolvingNeuralNet) network.setValue("synapseEndZ",musclePositions[endNeuron-numHidden][2]);
					if(evolvingNeuralNet) network.setValue("synapseEndL",1);
					//if(addDistanceFromShell) network.setValue("ds",e); //ACHEN: Shell distance
					//if(addInsideOrOutside) network.setValue("inout",e);
					network.setValue("Bias",0.3);                       
									
					network.update();                                   //JMC: on this line we run the CPPN network...  

					if (true)//network.getValue("thisSynapsePresence")>0)
					{
						synapseWeights[numSensors*numHidden + numHidden*numHidden + startNeuron*numMuscles + endNeuron] = network.getValue("thisSynapseWeight");
					}
					else
					{
						synapseWeights[numSensors*numHidden + numHidden*numHidden + startNeuron*numMuscles + endNeuron] = 0;
					}
				}
			}

			// for (int i=0; i<numSensors*numHidden+numHidden*(numHidden+numMuscles); i++)
			// {
			// 	cout << "weight for "<< i << ": " << synapseWeights[i] << endl;
			// }

			// for (int i=0; i<numSensors+numHidden+numMuscles; i++)
			// {
			// 	cout << "presynaptic bias for "<< i << ": " << nodeBias[i] << endl;
			// 	cout << "time constant for "<< i << ":    " << nodeTimeConstant[i] << endl;
			// }

			bool hasSynapse = false;
			for (int i=0; i<numSensors*numHidden; i++)
			{
				if (synapseWeights[i]!=0) {hasSynapse = true; break;}
			}
			if (not hasSynapse)
			{
				cout << "No syanapse from Sensors to Hidden Layer... assigning fitness of 0.00001" << endl;
				return 0.00001;
			}

			hasSynapse = false;
			for (int i=numSensors*numHidden+numHidden*numHidden; i<numSensors*numHidden+numHidden*numHidden+numHidden*numMuscles; i++)
			{
				if (synapseWeights[i]!=0) {hasSynapse = true; break;}
			}
			if (not hasSynapse)
			{
				cout << "No syanapse from Hidden Layer to Muscles... assigning fitness of 0.00001" << endl;
				return 0.00001;
			}

		}


        // for (int thisHidden; thisHidden<numHidden; thisHidden++)
        // {
        // 	cout << hiddenPositions[thisHidden][0] << ", " << hiddenPositions[thisHidden][1] << ", " << hiddenPositions[thisHidden][2] << endl; 
        // }

		/*
		nac: TODO!!!
		for each node:
			for each other node:
		{
			calc synapse weight between node and other node
		}
		*/
		// int vCounter = 0;
		// for (int z=0; z<num_z_voxels; z++)
		// {
		// 	for (int y=0; y<num_y_voxels; y++)
		// 	{
		// 		for (int x=0; x<num_x_voxels; x++)
		// 		{
		// 			cout << ArrayForVoxelyze[vCounter];
		// 			vCounter++;
		// 		}
		// 	}
		// 	cout << endl;
		// }


	
		float fitness;
		float origFitness = 0.00001;
		int voxelPenalty;
		int numTotalVoxels = 0;
		int numMuscleVoxels = 0;
		int numSensorVoxels = 0;
		for (int j=0; j<ContinuousArray.GetFullSize(); j++)  //iterate through each location of the continuous matrix
		{
			if ( ArrayForVoxelyze[j]>0 ) { numTotalVoxels++; }
			if ( SensorArrayForVoxelyze[j]>0 ) { numSensorVoxels++; }
			if ( MuscleArrayForVoxelyze[j]>0 ) { numMuscleVoxels++; }
		}

		if (numTotalVoxels<2) {cout << "No Voxels Filled... setting fitness to 0.00001" << endl; return 0.00001;}

		if (int(NEAT::Globals::getSingleton()->getParameterValue("PenaltyType")) == 3) 
		{
			cout << endl <<"ERROR: Connection Penalty Not Yet Implemented" << endl;
			exit(0);
			// voxelPenalty = numConnections/2;
		} else {
			if (int(NEAT::Globals::getSingleton()->getParameterValue("PenaltyType")) == 2) 
			{
				voxelPenalty = numMuscleVoxels;
			} else {
				voxelPenalty = numTotalVoxels;
			}
		}

		// int voxelPenalty = writeVoxelyzeFile(ContinuousArray, PassiveStiffArray, PassiveSoftArray, Phase1Array, Phase2Array, individualID);
		writeVoxelyzeFile(ArrayForVoxelyze, SensorArrayForVoxelyze, MuscleArrayForVoxelyze, individualID, synapseWeights, sensorPositions, hiddenPositions, musclePositions, numSensors, numHidden, numMuscles, nodeBias, nodeTimeConstant, individual);

//		if (saveVxaOnly)
//		{
//			std::ostringstream addFitnessToInputFileCmd;
//			addFitnessToInputFileCmd << "mv " << individualID << "_genome.vxa "<< genNum << "/" << individualID << "--adjFit_"<< individual.getFitness() << "--numFilled_" << individual.getNumFilled() << "--origFit_"<< individual.getOrigFitness() << "_genome.vxa";
//			int exitCode3 = std::system(addFitnessToInputFileCmd.str().c_str());
//			return 0.001;
//		}

		// if (numTotalVoxels < 2)
		// {
		// 	fitness = 0.00001;
		// 	cout << "individual had less than 2 voxels filled, skipping evaluation..." << endl;
		// 	//skippedEvals++;	
		// } else
		if (true)
		{
			//nac: check md5sum of vxa file	
			//std::ostringstream md5sumCmd;
			//md5sumCmd << "md5sum " << individualID << "_genome.vxa";
			//FILE* pipe = popen(md5sumCmd.str().c_str(), "r");

			FILE* pipe = popen("md5sum md5sumTMP.txt", "r");			
			if (!pipe) {cout << "ERROR 1, exiting." << endl << endl; return 0.00001;}
			char buffer[128];
			std::string result = "";
			while(!feof(pipe)) 
			{
				if(fgets(buffer, 128, pipe) != NULL)
					result += buffer;
			}
			pclose(pipe);

			std::string md5sumString = result.substr(0,32);
			// cout << "md5sum: " << md5sumString << endl;
			individual->setThismd5(md5sumString);

			cout << "got md5 for: " << individual->getThismd5() << " (born from: " << individual->getParent1md5() << " and " << individual->getParent2md5() << ")" << endl; 
			
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
				clock_t start;
				clock_t end;
				start = clock();
				// timeval tim;
				// gettimeofday(&tim, NULL);
				// double t1=tim.tv_sec+(tim.tv_usec*1000000.0);
				// cout << "t1" << t1 << endl;
				// cout << "t1_adj" << tim.tv_usec/1000000.0 << endl;
	
				//cout << "individual: " << individualID << endl;
		
				// cout << "starting voxelyze evaluation now" << endl;

				std::ostringstream callVoxleyzeCmd;
				// callVoxleyzeCmd << "../config/voxelize -f " << individualID << "_genome.vxa";
				callVoxleyzeCmd << "./voxelize -f " << individualID << "_genome.vxa";

				std::ostringstream outFileName;
				outFileName << individualID << "_fitness.xml";  

				fitness = 0.00001;

				// std::ifstream infile(outFileName.str().c_str());

				int exitCode;
		
				// int exitCode = std::system(callVoxleyzeCmd.str().c_str());

			// std::string sIdentifier;
			// pid_t pid = fork();
			// cout << "pid:" << pid << endl;
			// if(pid) 
			// {
			//     int status;
			//     float elapsedTime = 0.0;
			//     while(waitpid(0, &status, 0)!=0 and elapsedTime < 0.01) 
			//     {
			//         // the child process complete within 30 seconds
			//         end = clock();
			//         printf("voxelyze finished in %.6f seconds\n", float(end-start)/CLOCKS_PER_SEC);
			//         sleep(0.01);
			//         elapsedTime += 0.01;

			//     }
			//     if (not(waitpid(0, &status, 0)!=0 and waitpid(0, &status, 0)!=-1))
			//     {
			//         // the child process does not complete within 30 seconds
			//         kill(0, SIGTERM);
			//         cout << "voxelyze timed out at 60 seconds***********************************************************" << endl;
			//         return 0.0001;
			//     }
			// }
			// else 
			// {
			//     // child process
			//     cout << "starting voxelyze" << endl;
			//     exitCode = std::system(callVoxleyzeCmd.str().c_str());
			//     cout << "voxelyze finished" << endl;
			//     exit(1);
			// }
// 			   if (pID == 0)                // child
// 			   {
// 			      Code only executed by child process
// 			   	  cout << "starting voxelyze evaluation now" << endl;

// 			      sIdentifier = "Child Process: ";
// 			      int status;
// 			      execl(".","voxelize","-f",(individualID+"_genome.vxa").c_str());
// 			      execl(".",callVoxleyzeCmd.str().c_str());
// 			      execl(callVoxleyzeCmd.str().c_str());
// 			      while (-1 == waitpid(pID, &status, 0)) {cout<<"waiting..."<<endl;}
// 			      sleep(5);
// 			      exitCode = std::system(callVoxleyzeCmd.str().c_str());
// 			      exitCode = std::system(callVoxleyzeCmd.str().c_str());
// 			      cout << "voxelyze finished" << endl;
// 			      exit(1);
// 			    }
// 			    else if (pID < 0)            // failed to fork
// 			    {
// 			        cerr << "Failed to fork" << endl;
// 			        //return 0.00001;
// 			        exit(1);
// 			        // Throw exception
// 			    }
// 			    else                                   // parent
// 			    {
// 			      // Code only executed by parent process

// 			      // sIdentifier = "Parent Process:";
// 			    	int status;
// 					pid_t result = waitpid(ChildPID, &status, WNOHANG);
// 					if (result == 0) {
// 					  // Child still alive
// 					} else if (result == -1) {
// 					  // Error 
// 					} else {
// 					  // Child exited
// }

// 			    }

			    // Code executed by both parent and child.
			  
			    // cout << sIdentifier;

			    // sleep(5);
			    // exit(0);

                


				FILE* input = popen(callVoxleyzeCmd.str().c_str(),"r");
				// FILE *input = popen("sleep 300","r");

				//int exitCode2 = std::system("mv softbotsOutputTmp.xml softbotsOutput.xml");


				// gettimeofday(&tim, NULL);
				// double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
		
				// printf("voxelyze took %.6lf seconds\n", t2-t1);

				//cout<<"wrote the file and exiting!" << endl;

				//timespec time1, time2;

				//float startClock = clock_gettime(CLOCK_MONOTONIC, &time1);

		
				//float timeUsed = clock_gettime(CLOCK_MONOTONIC, &time2) - startClock;

				//cout << "Voxelyze used " << timeUsed << " seconds" << endl;

				// cout << "Exit Code From Voxelyze:" << WEXITSTATUS(exitCode) << endl;

				// TiXmlDocument doc;
 		// <-- for real
				//outFileName << "nan_fitness.xml";						// <-- debug only

				// if(!doc.LoadFile(outFileName.str().c_str()))
				// {
				// 	cerr << doc.ErrorDesc() << endl;
				// 	return 0.000001; //return FAILURE;
				// }	
		
				// TiXmlElement* root = doc.FirstChildElement();
				// if(root == NULL)
				// {
				// 	cerr << "Failed to load file: No root element."
				// 		 << endl;
				// 	doc.Clear();
				// 	return 0.001; //return FAILURE;
				// }
			// FILE *popenPsFile = popen("ps axu > ","r");
			// std::fstream psfile(popenPsFile);

			// int exitCode3 = std::system("ps axu > /tmp/HnPsFile.txt");
			// std::ifstream psfile("/tmp/HnPsFile.txt");
			// std::string thisLine;
			// if (psfile.is_open())
  	// 		{
			// 	while (std::getline(psfile, thisLine))
			// 	{
			// 		std::size_t foundvox = thisLine.find("./voxelize");
			// 		if (foundvox!=std::string::npos)
			// 		{
			// 			cout << atoi(thisLine.substr(foundvox-8,4).c_str()) << endl;
			// 			// if (thisLine.substr(foundvox-5,1) == "2")
			// 			// {
			// 			// 	std::size_t foundsp = thisLine.find(" ");
			// 			// 	cout << "killing process "<< atoi(thisLine.substr(foundsp,11).c_str()) << endl;
			// 			// 	int exitCode4 = std::system("kill "+atoi(thisLine.substr(foundsp,11).c_str()));
			// 			// }
			// 		}
			// 	}
			// }
			// exit(0);
			
	
double t2;
bool doneEval = false;
// std::cout << "waiting";
while (not doneEval)
{
	// gettimeofday(&tim, NULL);
	// t2=tim.tv_sec+(tim.tv_usec*1000000.0);
	end = clock();
	std::ifstream infile(outFileName.str().c_str());
	if (infile.is_open())
	{
		// cout << "done" << endl;
		// printf("voxelyze took %.6lf seconds\n", double(end-start)/CLOCKS_PER_SEC);
		printf("voxelyze took %.6lf seconds\n", float(end-start)/CLOCKS_PER_SEC);
		sleep(0.1);
		doneEval = true;

		// exit(0);
	}
	else
	{
		sleep(0.01);
		// std::cout << ".";
		if ( double(end-start)/CLOCKS_PER_SEC > 130.0)//num_x_voxels*num_y_voxels*num_z_voxels/10)
		{
			// cout << "voxelyze hung after " << double(end-start)/CLOCKS_PER_SEC  << " seconds... assigning fitness of 0.00001"<<endl<<endl;
			cout << "voxelyze hung after 120 seconds... assigning fitness of 0.00001"<<endl;
			// kill(0, SIGTERM);

			int exitCode3 = std::system("ps axu > /tmp/HnPsFile.txt");
			std::ifstream psfile("/tmp/HnPsFile.txt");
			std::string thisLine;
			if (psfile.is_open())
  			{
				while (std::getline(psfile, thisLine))
				{
					std::size_t foundvox = thisLine.find("./voxelize");
					if (foundvox!=std::string::npos)
					{
						if (atoi(thisLine.substr(foundvox-8,4).c_str()) >= 2)
						{
							std::size_t foundsp = thisLine.find(" ");
							cout << ("kill "+thisLine.substr(foundsp,11)).c_str() << endl;
							int exitCode4 = std::system(("kill "+thisLine.substr(foundsp,11)).c_str());
						}
					}
				}
			}

			cout << endl;
			return 0.00001;
			// exit(0);
		}
	}
	infile.close();
	// cout << "t2" << t2 << endl;
}
pclose(input);
				std::ifstream infile(outFileName.str().c_str());

				std::string line;
				float FinalCOM_DistX;
				float FinalCOM_DistY;
				float FinalCOM_DistZ;
				if (infile.is_open())
  				{
					while (std::getline(infile, line))
					{
					    // std::istringstream iss(line);
					    // cout << "line: " << line << endl;
					    // int a, b;
					    // if (!(iss >> a >> b)) { break; } // error

					    // process pair (a,b)
					    std::size_t foundx = line.find("<FinalCOM_DistX>");
					    if (foundx!=std::string::npos)
					    {
					    	FinalCOM_DistX = atof(line.substr(foundx+strlen("<FinalCOM_DistX>"),line.find("</")-(foundx+strlen("<FinalCOM_DistX>"))).c_str());
					    }
					    std::size_t foundy = line.find("<FinalCOM_DistY>");
					    if (foundy!=std::string::npos)
					    {
					    	FinalCOM_DistY = atof(line.substr(foundy+strlen("<FinalCOM_DistY>"),line.find("</")-(foundy+strlen("<FinalCOM_DistY>"))).c_str());
					    }
					    std::size_t foundz = line.find("<FinalCOM_DistZ>");
					    if (foundz!=std::string::npos)
					    {
					    	FinalCOM_DistZ = atof(line.substr(foundz+strlen("<FinalCOM_DistZ>"),line.find("</")-(foundz+strlen("<FinalCOM_DistZ>"))).c_str());
					    }
					}

					fitness = pow(pow(FinalCOM_DistX,2)+pow(FinalCOM_DistY,2),0.5);
					// cout << "Return from voxelyze: " << fitness << endl;
					fitness = fitness/(max(num_x_voxels,max(num_y_voxels,num_z_voxels))*VoxelSize);
					cout << "*** Fitness from voxelyze: " << fitness << endl;

					infile.close();
				}
				

		// 		for(TiXmlElement* elem1 = root->FirstChildElement(); elem1 != NULL; elem1 = elem1->NextSiblingElement())
		// 		{
			
		// 			for(TiXmlElement* elem = elem1->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
		// 			{

		// 				string elemName = elem->Value();
		// /*				cout << "Elem Name: " << elemName << endl;
	
		// 				const char* attr;
		// 				if(elemName == "Fitness")//"FinalCOM_Dist")
		// 				{
		// 					//cout << "got here 4" << endl;
		// 					attr = elem->Attribute("FinalCOM_Dist");
		// 					if(attr != NULL)
		// 					//cout<< "got here 3" << endl	; // Do stuff with it
							
		// 				}
		// */
			
		// 				// for(TiXmlNode* e = elem->FirstChild(); e != NULL; e = e->NextSibling())
		// 				// {

		// 				// 	// cout << "got here 1" << endl;
		// 				// 	TiXmlText* text = e->ToText();
		// 				// 	cout << "element from voxelyze: " << e->Value() << endl;
		// 				// 	cout << "raw fitness value from voxelyze: " << fabs(atof(e->Value())) << endl;
		// 				// 	cout << "body length fitness value from voxelyze: " << fabs(atof(e->Value()))/(num_x_voxels*VoxelSize) << endl;
		// 				// 	fitness = fabs(atof(e->Value()))/(num_x_voxels*VoxelSize);
							
		// 				// 	if(text != NULL)
		// 				// 	{
		// 				// 		cout << "text test 1: " << text->Value() << endl;
		// 				// 		//cout << "got here 5, text = NULL" << endl;
		// 				// 		//continue;
		// 				// 	//}
						
		// 				// 	if (std::string::npos != string(e->Value()).find('n'))
		// 				// 	{
		// 				// 	  fitness = 0.00001;
		// 				// 	  cout << "THIS INDIVIDUAL RETURNED NaN... setting arbitrary low fitness" << endl;
		// 				// 	}
		// 				// /*
		// 				// 	string t = text->Value();
		// 				// 	// Do stuff
		// 				// 	fitness = fabs(atof(t.c_str()))/(num_x_voxels*VoxelSize); 
		// 				// 	cout << "TEST FITNESS FROM ATOF: " << fitness << endl;
		// 				// 			//NOTE: taking abs of dispalcement (COM) for fitness (nac)
		// 				// 			//NOTE: using num body length in x direction as fitness (voxel size set at 0.001) (nac)
		// 				// 	cout << "got here 2" << endl;
		// 				// 	cout << "text = " << t << endl;

		// 				// 	}
		// 				// 	*/
		// 				// }
		// 			}
		// 		}
		// 		doc.Clear();
			}

			
			if (fitness < 0.00001) fitness = 0.00001;
			if (fitness > 1000) fitness = 0.00001;

			// cout << "Original fitness value (in body lengths): " << fitness << endl;
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
			// else {cout << "No penalty funtion, using original fitness" << endl;}

			// nac: update fitness lookup table		
			//FitnessRecord fitness, origFitness;
			//std::map<double, FitnessRecord> fits;

			if (fitness < 0.00001) fitness = 0.00001;
			if (fitness > 1000) fitness = 0.00001;

			pair <double, double> fits (fitness, origFitness);	
			fitnessLookup[md5sumString]=fits;
	
		}
		if (int(NEAT::Globals::getSingleton()->getParameterValue("PenaltyType")) != 0) {cout << "ADJUSTED FITNESS VALUE: " << fitness << endl;}

		std::ostringstream addFitnessToInputFileCmd;
		std::ostringstream addFitnessToOutputFileCmd;

		if (genNum % (int)NEAT::Globals::getSingleton()->getParameterValue("RecordEntireGenEvery")==0) // || bestFit<fitness) // <-- nac: individuals processed one at a time, not as generation group
		{		
			addFitnessToInputFileCmd << "mv " << individualID << "_genome.vxa Gen_"; 

			char buffer1 [100];
			sprintf(buffer1, "%04i", genNum);
			addFitnessToInputFileCmd << buffer1;

			addFitnessToInputFileCmd << "/" << individual->getThismd5() << "--adjFit_";

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
		
		if (fitness < 0.00001) fitness = 0.00001;	
		individual->setOrigFitness(origFitness);
		
			
		//individual.setOrigFitness(origFitness);     // nac: WHY DON'T THESE WORK??!!	
		//individual.setNumFilled(voxelPenalty);	
		
		return fitness; 
		//return 100;

    }
	

    void SoftbotsExperiment::processGroup(shared_ptr<NEAT::GeneticGeneration> generation)
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
			tmp << "Softbots--" << NEAT::Globals::getSingleton()->getOutputFilePrefix() << "--Gen_" ;
			char buffer [50];
			sprintf(buffer, "%04i", genNum);
			tmp << buffer;
			tmp << "--Ind_" << individual;
	  		string individualID = tmp.str();
			cout << endl << individualID << endl;
			

			#if SHAPES_EXPERIMENT_DEBUG
			  cout << "in SoftbotsExperiment.cpp::processIndividual, processing individual:  " << individual << endl;
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

			// cout << "Individual Evaluation complete!\n";
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

    void SoftbotsExperiment::processIndividualPostHoc(shared_ptr<NEAT::GeneticIndividual> individual)
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

    
    void SoftbotsExperiment::printNetworkCPPN(shared_ptr<const NEAT::GeneticIndividual> individual)
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

    Experiment* SoftbotsExperiment::clone()
    {
        SoftbotsExperiment* experiment = new SoftbotsExperiment(*this);

        return experiment;
    }
    
	
	bool SoftbotsExperiment::converged(int generation) {
		if(generation == convergence_step)
			return true;
		return false;
	}		

	void SoftbotsExperiment::writeAndTeeUpParentsOrderFile() {		//this lets django know what the order in the vector was of each parent for each org in the current gen
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

    // int SoftbotsExperiment::writeVoxelyzeFile( CArray3Df ContinuousArray, CArray3Df PassiveStiffArray, CArray3Df PassiveSoftArray,CArray3Df Phase1Array , CArray3Df Phase2Array, string individualID)
    int SoftbotsExperiment::writeVoxelyzeFile( CArray3Df ArrayForVoxelyze, CArray3Df SensorArrayForVoxelyze, CArray3Df MuscleArrayForVoxelyze, string individualID, float synapseWeights [100000], float sensorPositions [100][3], float hiddenPositions [100][3], float musclePositions [100][3], int numSensors, int numHidden, int numMuscles, float nodeBias [300], float nodeTimeConstant [300], shared_ptr<NEAT::GeneticIndividual> individual)
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
<SlowDampingZ>0.01</SlowDampingZ>\n\
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
<StopCondition>\n\
<StopConditionType>2</StopConditionType>\n\
<StopConditionValue>" << (float(NEAT::Globals::getSingleton()->getParameterValue("SimulationTime"))) << "</StopConditionValue>\n\
</StopCondition>\n\
<GA>\n\
<WriteFitnessFile>1</WriteFitnessFile>\n\
<FitnessFileName>" << individualID << "_fitness.xml</FitnessFileName>\n\
</GA>\n\
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
<TempPeriod>0.025</TempPeriod>\n";
		if (evolvingNeuralNet) {myfile << "<EvolvingNeuralNetworks>1</EvolvingNeuralNetworks>\n";}
		else {myfile << "<EvolvingNeuralNetworks>0</EvolvingNeuralNetworks>\n";}
		if (phaseDrivenActuation) {myfile << "<PhaseDrivenActuation>1</PhaseDrivenActuation>\n";}
		else {myfile << "<PhaseDrivenActuation>0</PhaseDrivenActuation>\n";}
		myfile << "\
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
<Palette>\n";
		myfile << "\
<Material ID=\"1\">\n\
<MatType>0</MatType>\n\
<Name>Soft</Name>\n\
<Display>\n\
<Red>0</Red>\n\
<Green>1</Green>\n\
<Blue>1</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<IsConductive>1</IsConductive>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>1e+007</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>2e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0.01</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n";
		myfile << "\
<Material ID=\"2\">\n\
<MatType>0</MatType>\n\
<Name>Hard</Name>\n\
<Display>\n\
<Red>0</Red>\n\
<Green>0</Green>\n\
<Blue>1</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<IsConductive>1</IsConductive>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>" << stiffness.str().c_str() << "e+006</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>2e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0.01</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n";
		myfile << "\
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
<IsConductive>1</IsConductive>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>1e+007</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>2e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>-0.01</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n";
		myfile << "\
<Material ID=\"4\">\n\
<MatType>0</MatType>\n\
<Name>Active_+</Name>\n\
<Display>\n\
<Red>1</Red>\n\
<Green>0</Green>\n\
<Blue>0</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<IsConductive>1</IsConductive>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>1e+007</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>2e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0.01</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n";
// <Material ID=\"5\">\n\
// <MatType>0</MatType>\n\
// <Name>Obstacle</Name>\n\
// <Display>\n\
// <Red>1</Red>\n\
// <Green>0.47</Green>\n\
// <Blue>1</Blue>\n\
// <Alpha>1</Alpha>\n\
// </Display>\n\
// <Mechanical>\n\
// <MatModel>0</MatModel>\n\
// <Elastic_Mod>1e+007</Elastic_Mod>\n\
// <Plastic_Mod>0</Plastic_Mod>\n\
// <Yield_Stress>0</Yield_Stress>\n\
// <FailModel>0</FailModel>\n\
// <Fail_Stress>0</Fail_Stress>\n\
// <Fail_Strain>0</Fail_Strain>\n\
// <Exclude_COM>1</Exclude_COM>\n\
// <Density>5e+07</Density>\n\
// <Poissons_Ratio>0.35</Poissons_Ratio>\n\
// <CTE>0.0</CTE>\n\
// <uStatic>1</uStatic>\n\
// <uDynamic>1</uDynamic>\n\
// </Mechanical>\n\
// </Material>\n";
		myfile << "\
</Palette>\n\
<Structure Compression=\"LONG_READABLE\">\n\
<X_Voxels>" << num_x_voxels+(2*ObsBuf) << "</X_Voxels>\n\
<Y_Voxels>" << num_y_voxels+(2*ObsBuf) << "</Y_Voxels>\n\
<Z_Voxels>" << num_z_voxels << "</Z_Voxels>\n\
<NumSensors>" << numSensors << "</NumSensors>\n\
<NumPacemaker>" << numPacemaker << "</NumPacemaker>\n\
<NumHidden>" << numHidden << "</NumHidden>\n\
<NumMuscles>" << numMuscles << "</NumMuscles>\n\
<Data>\n\
<Layer><![CDATA[";

// CArray3Df ArrayForVoxelyze = createArrayForVoxelyze(ContinuousArray, PassiveStiffArray, PassiveSoftArray, Phase1Array, Phase2Array);

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
			if ((y>=ObsBuf) && (y<ObsBuf+num_y_voxels) && (x>=ObsBuf) && (x<ObsBuf+num_x_voxels))
			{
				myfile << ArrayForVoxelyze[v] << ",";
				md5file << ArrayForVoxelyze[v];
				v++;
			}
			//if this is a place where we want an obstacle, place it
			else if ((DoObs > 0) && (z==0) && (
				((CurDist>ObsDistDes) && (CurDist<ObsDistDes+ObsThickness))	/*a first concentric circle*/
				|| ((CurDist>ObsDistDes*1.5) && (CurDist<ObsDistDes*1.5+ObsThickness))	/*a second concentric circle*/
				|| ((CurDist>ObsDistDes*2) && (CurDist<ObsDistDes*2+ObsThickness))	/*a third concentric circle*/
				))
			{
				myfile << 5;
				md5file << 5;
			}
			//o.w., no voxel is placed 
			else 
			{
				myfile << 0;
				md5file << 0;
			}

		}
	}
	if (z<num_z_voxels-1) {myfile << "]]></Layer>\n<Layer><![CDATA["; md5file << "\n";}
}

myfile << "]]></Layer>\n\
</Data>\n\
<SensorType>\n\
<Layer><![CDATA[";

v=0;
for (int z=0; z<num_z_voxels; z++)
{
	for (int y=0; y<num_y_voxels+(2*ObsBuf); y++)
	{
		for (int x=0; x<num_x_voxels+(2*ObsBuf); x++)
		{
			myfile << SensorArrayForVoxelyze[v] << ",";
			md5file << SensorArrayForVoxelyze[v];
			v++;
		}
	}
	if (z<num_z_voxels-1) {myfile << "]]></Layer>\n<Layer><![CDATA["; md5file << "\n";}
}

myfile << "]]></Layer>\n\
</SensorType>\n\
<MuscleType>\n\
<Layer><![CDATA[";

v=0;
for (int z=0; z<num_z_voxels; z++)
{
	for (int y=0; y<num_y_voxels+(2*ObsBuf); y++)
	{
		for (int x=0; x<num_x_voxels+(2*ObsBuf); x++)
		{
			myfile << MuscleArrayForVoxelyze[v] << ",";
			md5file << MuscleArrayForVoxelyze[v];
			v++;
		}
	}
	if (z<num_z_voxels-1) {myfile << "]]></Layer>\n<Layer><![CDATA["; md5file << "\n";}
}

myfile << "]]></Layer>\n\
</MuscleType>\n\
<NodeBias>\n\
<Layer><![CDATA[";

for (int i=0; i<numSensors+numHidden+numMuscles; i++)
{
	myfile << nodeBias[i] << ",";
	md5file << nodeBias[i];
}

myfile << "]]></Layer>\n\
</NodeBias>\n\
<SynapseWeight>\n\
<Layer><![CDATA[";

for (int i=0; i<(numSensors+numHidden+numMuscles)*numHidden; i++)
{
	myfile << synapseWeights[i] << ",";
	md5file << synapseWeights[i];
}
			
myfile << "]]></Layer>\n\
</SynapseWeight>\n\
<NodeTimeConstant>\n\
<Layer><![CDATA[";
for (int i=0; i<numSensors+numHidden+numMuscles; i++)
{
	myfile << nodeTimeConstant[i] << ",";
	md5file << nodeTimeConstant[i];
}		
myfile << "]]></Layer>\n\
</NodeTimeConstant>\n";


myfile<< "<SensorPositions>\n\
<Layer><![CDATA[";
for (int i=0; i<numSensors; i++)
{
	for (int j=0; j<3; j++)
	{
		myfile << sensorPositions[i][j] << ",";
		md5file << sensorPositions[i][j];
	}
}
myfile << "]]></Layer>\n\
</SensorPositions>\n";

myfile<< "<HiddenPositions>\n\
<Layer><![CDATA[";
for (int i=0; i<numHidden; i++)
{
	for (int j=0; j<3; j++)
	{
		myfile << hiddenPositions[i][j] << ",";
		md5file << hiddenPositions[i][j];
	}
}
myfile << "]]></Layer>\n\
</HiddenPositions>\n";


myfile<< "<MusclePositions>\n\
<Layer><![CDATA[";
for (int i=0; i<numMuscles; i++)
{
	for (int j=0; j<3; j++)
	{
		myfile << musclePositions[i][j] << ",";
		md5file << musclePositions[i][j];
	}
}
myfile << "]]></Layer>\n\
</MusclePositions>\n";


myfile << "</Structure>\n\
</VXC>\n\
</VXA>\n\
<!-- speciesID: " <<  individual->getSpeciesID() << " -->\n\
<!-- parent1md5: " <<  individual->getParent1md5() << " -->\n\
<!-- parent2md5: " <<  individual->getParent2md5() << " -->\n\
<!-- nodesAdded: " <<  individual->getNodesAdded() << " -->\n\
<!-- linksAdded: " <<  individual->getLinksAdded() << " -->\n\
<!-- linkWeightsMutated: " <<  individual->getLinkWeightsMutated() << " -->\n\
<!-- linkWeightsNotMutated: " <<  individual->getLinkWeightsNotMutated() << " -->\n\
<!-- linksDemolished: " <<  individual->getLinksDemolished() << " -->\n\
";

  		myfile.close();
		md5file.close();
		
		// check if "robot" is empty space:
		for (int i=0; i<ArrayForVoxelyze.GetFullSize(); i++)
		{
			if (!ArrayForVoxelyze[i]==0)
			{
				numVoxelsFilled++;
			}

			if (evolvingSensorsAndMotors)
			{
				if (MuscleArrayForVoxelyze[v] > 0 )
				{
					numVoxelsActuated++;
				}
			}
			else
			{	
				if (ArrayForVoxelyze[i] > 2 )
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

	CArray3Df SoftbotsExperiment::createArrayForVoxelyze(CArray3Df ContinuousArray,CArray3Df PassiveStiffArray, CArray3Df PassiveSoftArray,CArray3Df Phase1Array,CArray3Df Phase2Array,CArray3Df PhaseSensorArray,CArray3Df PhaseMotorArray)
	{	
		// cout << "got to createArrayForVoxelyze" << endl;
		CArray3Df ArrayForVoxelyze(num_x_voxels, num_y_voxels, num_z_voxels); //array denoting type of material

		numVoxelsFilled   = 0;
		numVoxelsActuated = 0;
		numConnections    = 0;
		int v=0;
		// std::string materialTypes = NEAT::Globals::getSingleton()->getMaterialTypes();
		// bool motorOn = materialTypes.find("Motor")!=std::string::npos;

		for (int z=0; z<num_z_voxels; z++)
		{
			for (int y=0; y<num_y_voxels; y++)
			{
				for (int x=0; x<num_x_voxels; x++)
				{
					if (evolvingSensorsAndMotors)
					{
						if(ContinuousArray[v] < 0)
							{ ArrayForVoxelyze[v] = 0;}
						else
						{
							// if (PassiveSoftArray[v] >= PassiveStiffArray[v])
							// 	{ ArrayForVoxelyze[v] = 1;}
							// else
							// 	{ ArrayForVoxelyze[v] = 2;}
							// double maxVal = std::max(std::max(PassiveSoftArray[v],PassiveStiffArray[v]),std::max(Phase1Array[v],Phase2Array[v]));
							double maxVal = std::max(PassiveSoftArray[v],PassiveStiffArray[v]);
							// if (Phase2Array[v] == maxVal) {ArrayForVoxelyze[v] = 4;}
							// if (Phase1Array[v] == maxVal) {ArrayForVoxelyze[v] = 3;}
							if (PassiveStiffArray[v] == maxVal) {ArrayForVoxelyze[v] = 2;}
							if (PassiveSoftArray[v] == maxVal) {ArrayForVoxelyze[v] = 1;}
						}
					}

					else // notEvolvingSensorsAndMotors == muscle phases
					{
						if(ContinuousArray[v] < 0)
							{ ArrayForVoxelyze[v] = 0;}
						else
						{
							// if (PassiveSoftArray[v] >= PassiveStiffArray[v])
							// 	{ ArrayForVoxelyze[v] = 1;}
							// else
							// 	{ ArrayForVoxelyze[v] = 2;}
							double maxVal = std::max(std::max(PassiveSoftArray[v],PassiveStiffArray[v]),std::max(Phase1Array[v],Phase2Array[v]));
							if (Phase2Array[v] == maxVal) {ArrayForVoxelyze[v] = 4;}
							if (Phase1Array[v] == maxVal) {ArrayForVoxelyze[v] = 3;}
							if (PassiveStiffArray[v] == maxVal) {ArrayForVoxelyze[v] = 2;}
							if (PassiveSoftArray[v] == maxVal) {ArrayForVoxelyze[v] = 1;}
							// cout << "maxVal: " << maxVal << endl;
							// cout << "Phase2Array[v]: " << Phase2Array[v] << endl;
							// cout << "Phase1Array[v]: " << Phase1Array[v] << endl;
							// cout << "PassiveStiffArray[v]: " << PassiveStiffArray[v] << endl;
							// cout << "PassiveSoftArray[v]: " << PassiveSoftArray[v] << endl;
							// cout << "ArrayForVoxelyze[v]: " << ArrayForVoxelyze[v] << endl;
							// cout << endl;
						}
					}
					// // NOTE: this uses mateiral 0 as passive, material 1 as active phase 1, and material 2 as active phase 2
					// if (evolvingSensorsAndMotors)
					// {
					// 	if(ContinuousArray[v] < 0)
					// 	{ ArrayForVoxelyze[v] = 0;}
					// 	if (motorOn and PhaseMotorArray)
					// } else 
					// {

					// if (PhaseMotorArray[v]>0) 
					// {
					// 	ArrayForVoxelyze[v] = -1;
					// }
					// else
					// {
					// 	ArrayForVoxelyze[v] = 0;
					// }
					// if (PhaseMotorArray[v]<-0.9) 
					// {
					// 	ArrayForVoxelyze[v] = -1;
					// } 
/*				
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
*/		 
					// change "=" to random assignment? currently biases passive, then phase 1, then phase 2 if all equal
					// }

					v++;
				}
			}
		}

		ArrayForVoxelyze = makeOneShapeOnly(ArrayForVoxelyze);

		return ArrayForVoxelyze;
	}

	CArray3Df SoftbotsExperiment::createSensorArrayForVoxelyze(CArray3Df PhaseSensorArray,CArray3Df ArrayForVoxelyze)
	{	
		CArray3Df SensorArrayForVoxelyze (num_x_voxels, num_y_voxels, num_z_voxels); //array denoting type of material

		// numVoxelsFilled   = 0;
		// numVoxelsActuated = 0;
		// numConnections    = 0;
		int v=0;
		// std::string materialTypes = NEAT::Globals::getSingleton()->getMaterialTypes();
		// bool motorOn = materialTypes.find("Motor")!=std::string::npos;

		for (int z=0; z<num_z_voxels; z++)
		{
			for (int y=0; y<num_y_voxels; y++)
			{
				for (int x=0; x<num_x_voxels; x++)
				{
					// // NOTE: this uses mateiral 0 as passive, material 1 as active phase 1, and material 2 as active phase 2
					// if (evolvingSensorsAndMotors)
					// {
					// 	if(ContinuousArray[v] < 0)
					// 	{ ArrayForVoxelyze[v] = 0;}
					// 	if (motorOn and PhaseMotorArray)
					// } else 
					// {
					if (ArrayForVoxelyze[v] > 0)
					{
						if (PhaseSensorArray[v]>0) 
						{
							SensorArrayForVoxelyze[v] = -1;
						}
					}
					else
					{
						SensorArrayForVoxelyze[v] = 0;
					}
					// if (PhaseMotorArray[v]<-0.9) 
					// {
					// 	ArrayForVoxelyze[v] = -1;
					// } 
/*				
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
*/		 
					// change "=" to random assignment? currently biases passive, then phase 1, then phase 2 if all equal
					// }

					v++;
				}
			}
		}

		// ArrayForVoxelyze = makeOneShapeOnly(ArrayForVoxelyze);

		return SensorArrayForVoxelyze;
	}

	CArray3Df SoftbotsExperiment::makeOneShapeOnly(CArray3Df ArrayForVoxelyze)
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

	CArray3Df SoftbotsExperiment::numberSensors(CArray3Df MuscleArrayForVoxelyze)
	{
		bool fullNeighborhood = true;

		int muscleGroupIndex = 1;
		for (int zIndex=0; zIndex<num_z_voxels; zIndex++)
		{
			for (int yIndex=0; yIndex<num_y_voxels; yIndex++)
			{
				for (int xIndex=0; xIndex<num_x_voxels; xIndex++)
				{
					if (MuscleArrayForVoxelyze[coordinatesToIndex(xIndex,yIndex,zIndex)] == -1)
					{
						queue<int> queueToCheck;
						queueToCheck.push(coordinatesToIndex(xIndex,yIndex,zIndex));
						// list<int> alreadyChecked;
						// pair<  queue<int>, list<int> > queueAndList;
						// int startVoxel = int(MuscleArrayForVoxelyze.GetFullSize()/2);
						// queueToCheck.push(startVoxel);
						// alreadyChecked.push_back(startVoxel);
						int voxelToCircle;
						vector<int> coor;
						while (!queueToCheck.empty())
						{
//							MuscleArrayForVoxelyze[voxelToCircle] = muscleGroupIndex;
							voxelToCircle = queueToCheck.front();
							queueToCheck.pop();
							coor = indexToCoordinates(voxelToCircle);
//cout << "checking:" << coordinatesToIndex(coor[0],coor[1],coor[2]) << endl;
//cout << "checking: " << coor[0] << ", " << coor[1] << ", " << coor[2] << endl;
							if (fullNeighborhood)
							{
								for (int dx = -1; dx<=1; dx++)
								{
									for (int dy = -1; dy<=1; dy++)
									{
										for (int dz = -1; dz<=1; dz++)
										{
											if ( not (coor[0]+dx < 0 or coor[0]+dx >= num_x_voxels or coor[1]+dy < 0 or coor[1]+dy >= num_y_voxels or coor[2]+dz < 0 or coor[2]+dz >= num_z_voxels) )
											{
												if (MuscleArrayForVoxelyze[coordinatesToIndex(coor[0]+dx,coor[1]+dy,coor[2]+dz)] == -1)
												{
													MuscleArrayForVoxelyze[coordinatesToIndex(coor[0]+dx,coor[1]+dy,coor[2]+dz)] = muscleGroupIndex;
													queueToCheck.push(coordinatesToIndex(coor[0]+dx,coor[1]+dy,coor[2]+dz));
												}
											}
										}
									}
								}
							}
							else
							{
								for (int dx = -1; dx<=1; dx++)
								{
									if ( not (coor[0]+dx < 0 or coor[0]+dx >= num_x_voxels) )
									{
										if (MuscleArrayForVoxelyze[coordinatesToIndex(coor[0]+dx,coor[1],coor[2])] == -1)
										{
											MuscleArrayForVoxelyze[coordinatesToIndex(coor[0]+dx,coor[1],coor[2])] = muscleGroupIndex;
											queueToCheck.push(coordinatesToIndex(coor[0]+dx,coor[1],coor[2]));
										}
									}
								}
								for (int dy = -1; dy<=1; dy++)
								{
									if ( not (coor[1]+dy < 0 or coor[1]+dy >= num_y_voxels) )
									{
										if (MuscleArrayForVoxelyze[coordinatesToIndex(coor[0],coor[1]+dy,coor[2])] == -1)
										{
											MuscleArrayForVoxelyze[coordinatesToIndex(coor[0],coor[1]+dy,coor[2])] = muscleGroupIndex;
											queueToCheck.push(coordinatesToIndex(coor[0],coor[1]+dy,coor[2]));
										}
									}
								}
								for (int dz = -1; dz<=1; dz++)
								{
									if ( not (coor[2]+dz < 0 or coor[2]+dz >= num_z_voxels) )
									{
										if (MuscleArrayForVoxelyze[coordinatesToIndex(coor[0],coor[1],coor[2]+dz)] == -1)
										{
											MuscleArrayForVoxelyze[coordinatesToIndex(coor[0],coor[1],coor[2]+dz)] = muscleGroupIndex;
											queueToCheck.push(coordinatesToIndex(coor[0],coor[1],coor[2]+dz));
										}
									}
								}
							}
						}
						
						muscleGroupIndex++;

					}

				}
			}
		}
		return MuscleArrayForVoxelyze;
	}

	pair< queue<int>, list<int> > SoftbotsExperiment::circleOnce(CArray3Df ArrayForVoxelyze, queue<int> queueToCheck, list<int> alreadyChecked)
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

	int SoftbotsExperiment::leftExists(CArray3Df ArrayForVoxelyze, int currentPosition)
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

	int SoftbotsExperiment::rightExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (currentPosition % num_x_voxels == num_x_voxels-1) return -999;				
				else if (ArrayForVoxelyze[currentPosition + 1] == 0) return -999;
				else {numConnections++; return (currentPosition + 1);}
	}
			
	int SoftbotsExperiment::forwardExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (int(currentPosition/num_x_voxels) % num_y_voxels == num_y_voxels-1) return -999;				
				else if (ArrayForVoxelyze[currentPosition + num_x_voxels] == 0) return -999;
				else {numConnections++; return (currentPosition + num_x_voxels);}
	}

	int SoftbotsExperiment::backExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (int(currentPosition/num_x_voxels) % num_y_voxels == 0) return -999;				
				else if (ArrayForVoxelyze[currentPosition - num_x_voxels] == 0) return -999;
				else {numConnections++; return (currentPosition - num_x_voxels);}
	}

	int SoftbotsExperiment::upExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (int(currentPosition/(num_x_voxels*num_y_voxels)) % num_z_voxels == num_z_voxels-1) return -999;				
				else if (ArrayForVoxelyze[currentPosition + num_x_voxels*num_y_voxels] == 0) return -999;
				else {numConnections++; return (currentPosition + num_x_voxels*num_y_voxels);}
	}

	int SoftbotsExperiment::downExists(CArray3Df ArrayForVoxelyze, int currentPosition)
	{
				if (int(currentPosition/(num_x_voxels*num_y_voxels)) % num_z_voxels == 0) return -999;				
				else if (ArrayForVoxelyze[currentPosition - num_x_voxels*num_y_voxels] == 0) return -999;
				else {numConnections++; return (currentPosition - num_x_voxels*num_y_voxels);}
	}

	bool SoftbotsExperiment::isNextTo(int posX1, int posY1, int posZ1, int posX2, int posY2, int posZ2)
	{
		// bool nextTo = true;
		if (abs(posX1 - posX2) >1 ) {return false;}
		if (abs(posY1 - posY2) >1 ) {return false;}
		if (abs(posZ1 - posZ2) >1 ) {return false;}
		return true;
	}

	vector<int> SoftbotsExperiment::indexToCoordinates(int v)
	{
		int z = v/(num_x_voxels*num_y_voxels);
		int y = (v-z*num_x_voxels*num_y_voxels)/num_x_voxels;
		int x = v-z*num_x_voxels*num_y_voxels-y*num_x_voxels;
		int tempCoordinates[] = {x,y,z};
		vector<int> coordinates (tempCoordinates,tempCoordinates+3);
		return coordinates;
	}

	int SoftbotsExperiment::coordinatesToIndex(int x, int y, int z)
	{
		return x+y*num_x_voxels+z*num_x_voxels*num_y_voxels;
	}

}
