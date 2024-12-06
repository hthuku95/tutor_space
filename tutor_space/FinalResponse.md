(env) harry@DESKTOP-FTOKMKB:/mnt/d/projects/DevThukuDotIo/TutorSpace/tutor_space$ nano service-definition.json
(env) harry@DESKTOP-FTOKMKB:/mnt/d/projects/DevThukuDotIo/TutorSpace/tutor_space$ aws ecs create-service --cli-input-json file://service-definition.json
{
    "service": {
        "serviceArn": "arn:aws:ecs:eu-west-1:396913742200:service/tutorspace-cluster/tutorspace-service",
        "serviceName": "tutorspace-service",
        "clusterArn": "arn:aws:ecs:eu-west-1:396913742200:cluster/tutorspace-cluster",
        "loadBalancers": [
            {
                "targetGroupArn": "arn:aws:elasticloadbalancing:eu-west-1:396913742200:targetgroup/tutorspace-tg/795221176455e42b",
                "containerName": "tutorspace-backend",
                "containerPort": 8000
            }
        ],
        "serviceRegistries": [],
        "status": "ACTIVE",
        "desiredCount": 1,
        "runningCount": 0,
        "pendingCount": 0,
        "launchType": "FARGATE",
        "platformVersion": "LATEST",
        "platformFamily": "Linux",
        "taskDefinition": "arn:aws:ecs:eu-west-1:396913742200:task-definition/tutorspace-task:1",
        "deploymentConfiguration": {
            "deploymentCircuitBreaker": {
                "enable": false,
                "rollback": false
            },
            "maximumPercent": 200,
            "minimumHealthyPercent": 100
        },
        "deployments": [
            {
                "id": "ecs-svc/8010098638150428715",
                "status": "PRIMARY",
                "taskDefinition": "arn:aws:ecs:eu-west-1:396913742200:task-definition/tutorspace-task:1",
                "desiredCount": 0,
                "pendingCount": 0,
                "runningCount": 0,
                "failedTasks": 0,
                "createdAt": 1733318615.211,
                "updatedAt": 1733318615.211,
                "launchType": "FARGATE",
                "platformVersion": "1.4.0",
                "platformFamily": "Linux",
                "networkConfiguration": {
                    "awsvpcConfiguration": {
                        "subnets": [
                            "subnet-0b93a619af7015f61",
                            "subnet-087b9086804886e58",
                            "subnet-002e215a95f9e8af4"
                        ],
                        "securityGroups": [
                            "sg-0e839c0e4df8f6f81"
                        ],
                        "assignPublicIp": "ENABLED"
                    }
                },
                "rolloutState": "IN_PROGRESS",
                "rolloutStateReason": "ECS deployment ecs-svc/8010098638150428715 in progress."
            }
        ],
        "roleArn": "arn:aws:iam::396913742200:role/aws-service-role/ecs.amazonaws.com/AWSServiceRoleForECS",
        "events": [],
        "createdAt": 1733318615.211,
        "placementConstraints": [],
        "placementStrategy": [],
        "networkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": [
                    "subnet-0b93a619af7015f61",
                    "subnet-087b9086804886e58",
                    "subnet-002e215a95f9e8af4"
                ],
                "securityGroups": [
                    "sg-0e839c0e4df8f6f81"
                ],
                "assignPublicIp": "ENABLED"
            }
        },
        "healthCheckGracePeriodSeconds": 0,
        "schedulingStrategy": "REPLICA",
        "deploymentController": {
            "type": "ECS"
        },
        "createdBy": "arn:aws:iam::396913742200:root",
        "enableECSManagedTags": false,
        "propagateTags": "NONE",
        "enableExecuteCommand": false
    }
}
(env) harry@DESKTOP-FTOKMKB:/mnt/d/projects/DevThukuDotIo/TutorSpace/tutor_space$