"""
Security Stack - IAM Roles, Policies, Secrets Manager
"""
from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    aws_iam as iam,
    aws_secretsmanager as secretsmanager,
    aws_kms as kms,
    CfnOutput,
)


class SecurityStack(cdk.Stack):
    """Security infrastructure stack."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        environment: str,
        project_name: str,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.environment = environment
        self.project_name = project_name
        
        # ======================================================================
        # KMS Key for Encryption
        # ======================================================================
        self.kms_key = kms.Key(
            self,
            "EncryptionKey",
            alias=f"{project_name}-{environment}-key",
            description=f"KMS key for {project_name} {environment}",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.RETAIN if environment == "production" else cdk.RemovalPolicy.DESTROY,
        )
        
        # ======================================================================
        # Secrets Manager - API Keys
        # ======================================================================
        self.openai_api_key_secret = secretsmanager.Secret(
            self,
            "OpenAIApiKey",
            secret_name=f"{project_name}/{environment}/openai-api-key",
            description="OpenAI API Key",
            encryption_key=self.kms_key,
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"api_key": ""}',
                generate_string_key="placeholder",
                exclude_punctuation=True,
            ),
        )
        
        self.anthropic_api_key_secret = secretsmanager.Secret(
            self,
            "AnthropicApiKey",
            secret_name=f"{project_name}/{environment}/anthropic-api-key",
            description="Anthropic API Key",
            encryption_key=self.kms_key,
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"api_key": ""}',
                generate_string_key="placeholder",
                exclude_punctuation=True,
            ),
        )
        
        # ======================================================================
        # ECS Execution Role
        # ======================================================================
        self.ecs_execution_role = iam.Role(
            self,
            "ECSExecutionRole",
            role_name=f"{project_name}-{environment}-ecs-execution-role",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="ECS Task Execution Role",
        )
        
        self.ecs_execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
        )
        
        # Allow reading secrets
        self.ecs_execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadSecrets",
                effect=iam.Effect.ALLOW,
                actions=[
                    "secretsmanager:GetSecretValue",
                ],
                resources=[
                    self.openai_api_key_secret.secret_arn,
                    self.anthropic_api_key_secret.secret_arn,
                ],
            )
        )
        
        # Allow KMS decrypt
        self.kms_key.grant_decrypt(self.ecs_execution_role)
        
        # ======================================================================
        # ECS Task Role
        # ======================================================================
        self.ecs_task_role = iam.Role(
            self,
            "ECSTaskRole",
            role_name=f"{project_name}-{environment}-ecs-task-role",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="ECS Task Role with Bedrock access",
        )
        
        # Bedrock access policy
        self.ecs_task_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockFullAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:ListFoundationModels",
                    "bedrock:GetFoundationModel",
                    "bedrock-agent:InvokeAgent",
                    "bedrock-agent:Retrieve",
                    "bedrock-agent:RetrieveAndGenerate",
                    "bedrock-agent-runtime:InvokeAgent",
                    "bedrock-agent-runtime:Retrieve",
                    "bedrock-agent-runtime:RetrieveAndGenerate",
                ],
                resources=["*"],
            )
        )
        
        # S3 access policy (will be refined in storage stack)
        self.ecs_task_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3Access",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:DeleteObject",
                ],
                resources=[
                    f"arn:aws:s3:::{project_name}-{environment}-*",
                    f"arn:aws:s3:::{project_name}-{environment}-*/*",
                ],
            )
        )
        
        # DynamoDB access
        self.ecs_task_role.add_to_policy(
            iam.PolicyStatement(
                sid="DynamoDBAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                    "dynamodb:BatchGetItem",
                    "dynamodb:BatchWriteItem",
                ],
                resources=[
                    f"arn:aws:dynamodb:*:*:table/{project_name}-{environment}-*",
                ],
            )
        )
        
        # CloudWatch access
        self.ecs_task_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "cloudwatch:PutMetricData",
                ],
                resources=["*"],
            )
        )
        
        # X-Ray access
        self.ecs_task_role.add_to_policy(
            iam.PolicyStatement(
                sid="XRayAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                ],
                resources=["*"],
            )
        )
        
        # ======================================================================
        # Bedrock Agent Execution Role
        # ======================================================================
        self.bedrock_execution_role = iam.Role(
            self,
            "BedrockAgentRole",
            role_name=f"{project_name}-{environment}-bedrock-agent-role",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            description="Bedrock Agent Execution Role",
        )
        
        self.bedrock_execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockModelAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/anthropic.*",
                    "arn:aws:bedrock:*::foundation-model/amazon.*",
                    "arn:aws:bedrock:*::foundation-model/meta.*",
                    "arn:aws:bedrock:*::foundation-model/mistral.*",
                    "arn:aws:bedrock:*::foundation-model/cohere.*",
                ],
            )
        )
        
        self.bedrock_execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3KnowledgeBaseAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:ListBucket",
                ],
                resources=[
                    f"arn:aws:s3:::{project_name}-{environment}-*",
                    f"arn:aws:s3:::{project_name}-{environment}-*/*",
                ],
            )
        )
        
        # ======================================================================
        # Lambda Execution Role
        # ======================================================================
        self.lambda_execution_role = iam.Role(
            self,
            "LambdaExecutionRole",
            role_name=f"{project_name}-{environment}-lambda-execution-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Lambda Execution Role",
        )
        
        self.lambda_execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole")
        )
        
        # Bedrock access for Lambda
        self.lambda_execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock-agent-runtime:InvokeAgent",
                ],
                resources=["*"],
            )
        )
        
        # ======================================================================
        # GitHub Actions OIDC Provider and Role
        # ======================================================================
        github_oidc_provider = iam.OpenIdConnectProvider(
            self,
            "GitHubOIDCProvider",
            url="https://token.actions.githubusercontent.com",
            client_ids=["sts.amazonaws.com"],
        )
        
        self.github_actions_role = iam.Role(
            self,
            "GitHubActionsRole",
            role_name=f"{project_name}-{environment}-github-actions-role",
            assumed_by=iam.FederatedPrincipal(
                github_oidc_provider.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                    },
                    "StringLike": {
                        # Replace with your GitHub org/repo
                        "token.actions.githubusercontent.com:sub": "repo:${GITHUB_ORG}/${GITHUB_REPO}:*",
                    },
                },
                assume_role_action="sts:AssumeRoleWithWebIdentity",
            ),
            description="Role for GitHub Actions deployments",
            max_session_duration=cdk.Duration.hours(1),
        )
        
        # GitHub Actions permissions
        self.github_actions_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryPowerUser")
        )
        
        self.github_actions_role.add_to_policy(
            iam.PolicyStatement(
                sid="ECSDeployment",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecs:UpdateService",
                    "ecs:DescribeServices",
                    "ecs:DescribeClusters",
                    "ecs:DescribeTaskDefinition",
                    "ecs:RegisterTaskDefinition",
                    "ecs:DeregisterTaskDefinition",
                ],
                resources=["*"],
            )
        )
        
        self.github_actions_role.add_to_policy(
            iam.PolicyStatement(
                sid="CDKDeployment",
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudformation:*",
                    "ssm:GetParameter",
                    "sts:AssumeRole",
                ],
                resources=["*"],
            )
        )
        
        # ======================================================================
        # Outputs
        # ======================================================================
        CfnOutput(self, "ECSExecutionRoleArn", value=self.ecs_execution_role.role_arn)
        CfnOutput(self, "ECSTaskRoleArn", value=self.ecs_task_role.role_arn)
        CfnOutput(self, "BedrockExecutionRoleArn", value=self.bedrock_execution_role.role_arn)
        CfnOutput(self, "GitHubActionsRoleArn", value=self.github_actions_role.role_arn)
        CfnOutput(self, "KMSKeyArn", value=self.kms_key.key_arn)

