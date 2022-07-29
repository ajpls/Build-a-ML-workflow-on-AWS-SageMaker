# Build a ML Workflow On Amazon SageMaker For Scones Unlimited 

In this project, an image classification model was built with AWS SageMaker for Scones Unlimited. The model automatically detects types of delivery vehicles (bicycles and motorcycles) in order to route them to the correct loading bay and orders. The system can assist Scones Unlimited to optimize their operations by assigning near and far orders to delivery professionals with a bicycle and motorcycle, respectively. 

## Aims

The aims were to deploy a model that can scale to meet demand, with safeguards to monitor and control for drift or degraded performance. 

The project demonstrates building a ML Workflow with AWS applications such as:

1. AWS SageMaker to build an image classification model that can classify bicycles and motorcycles. 
2. AWS Endpoints for model deployment. 
3. AWS Lambda functions to build supporting services. 
4. AWS Step functions to compose the model and services into an event-driven application. 

## Project Steps Overview  

1. Data staging
2. Model training and deployment
3. Lambdas and step function workflow
4. Testing and evaluation
5. Cleanup cloud resources [1]

## References

[1] https://classroom.udacity.com/nanodegrees/nd189/parts/cd0386/modules/0d517ca6-b2cb-4231-a7bb-2e612b0826b9/lessons/3f8e8736-9f6a-4c45-aed3-84a33e64e91c/concepts/21cdc795-742f-458e-a996-e099ef204b46

