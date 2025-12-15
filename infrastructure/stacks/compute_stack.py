"""
Compute Stack - ECS Fargate, Lambda, API Gateway
"""
from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_iam as iam,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_logs as logs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_certificatemanager as acm,
    CfnOutput,
)


class ComputeStack(cdk.Stack):
    """Compute infrastructure stack."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        environment: str,
        project_name: str,
        vpc: ec2.IVpc,
        execution_role: iam.IRole,
        task_role: iam.IRole,
        data_bucket: s3.IBucket,
        bedrock_agent_id: str,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.environment = environment
        self.project_name = project_name
        is_production = environment == "production"
        
        # ======================================================================
        # ECS Cluster
        # ======================================================================
        self.ecs_cluster = ecs.Cluster(
            self,
            "ECSCluster",
            cluster_name=f"aibom-cluster-{environment}",
            vpc=vpc,
            container_insights=True,
            enable_fargate_capacity_providers=True,
        )
        
        # ======================================================================
        # ECS Task Definition
        # ======================================================================
        task_definition = ecs.FargateTaskDefinition(
            self,
            "TaskDefinition",
            family=f"{project_name}-{environment}",
            cpu=1024 if is_production else 512,
            memory_limit_mib=2048 if is_production else 1024,
            execution_role=execution_role,
            task_role=task_role,
        )
        
        # Main container
        container = task_definition.add_container(
            "AppContainer",
            container_name="app",
            image=ecs.ContainerImage.from_registry(
                f"{self.account}.dkr.ecr.{self.region}.amazonaws.com/{project_name}:latest"
            ),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="app",
                log_retention=logs.RetentionDays.ONE_MONTH,
            ),
            environment={
                "ENVIRONMENT": environment,
                "AWS_DEFAULT_REGION": self.region,
                "LOG_LEVEL": "INFO" if is_production else "DEBUG",
                "DATA_BUCKET": data_bucket.bucket_name,
                "BEDROCK_AGENT_ID": bedrock_agent_id,
            },
            secrets={
                # Secrets will be injected from Secrets Manager
                # "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(...),
                # "ANTHROPIC_API_KEY": ecs.Secret.from_secrets_manager(...),
            },
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
                interval=cdk.Duration.seconds(30),
                timeout=cdk.Duration.seconds(10),
                retries=3,
                start_period=cdk.Duration.seconds(60),
            ),
        )
        
        container.add_port_mappings(
            ecs.PortMapping(
                container_port=8080,
                protocol=ecs.Protocol.TCP,
            )
        )
        
        # X-Ray sidecar container
        xray_container = task_definition.add_container(
            "XRayDaemon",
            container_name="xray-daemon",
            image=ecs.ContainerImage.from_registry("amazon/aws-xray-daemon:latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="xray",
                log_retention=logs.RetentionDays.ONE_WEEK,
            ),
            cpu=32,
            memory_limit_mib=256,
        )
        
        xray_container.add_port_mappings(
            ecs.PortMapping(
                container_port=2000,
                protocol=ecs.Protocol.UDP,
            )
        )
        
        # ======================================================================
        # Application Load Balancer & Fargate Service
        # ======================================================================
        self.ecs_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "FargateService",
            cluster=self.ecs_cluster,
            service_name=f"aibom-service-{environment}",
            task_definition=task_definition,
            desired_count=2 if is_production else 1,
            public_load_balancer=True,
            assign_public_ip=False,
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            enable_execute_command=True,  # For debugging
            circuit_breaker=ecs.DeploymentCircuitBreaker(
                rollback=True,
            ),
            deployment_controller=ecs.DeploymentController(
                type=ecs.DeploymentControllerType.ECS,
            ),
            min_healthy_percent=50 if is_production else 0,
            max_healthy_percent=200,
        )
        
        # Configure health check
        self.ecs_service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=cdk.Duration.seconds(30),
            timeout=cdk.Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )
        
        # Auto-scaling
        scaling = self.ecs_service.service.auto_scale_task_count(
            min_capacity=1 if not is_production else 2,
            max_capacity=10 if is_production else 4,
        )
        
        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=cdk.Duration.seconds(60),
            scale_out_cooldown=cdk.Duration.seconds(60),
        )
        
        scaling.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=80,
            scale_in_cooldown=cdk.Duration.seconds(60),
            scale_out_cooldown=cdk.Duration.seconds(60),
        )
        
        # ======================================================================
        # API Gateway
        # ======================================================================
        self.api_gateway = apigw.RestApi(
            self,
            "ApiGateway",
            rest_api_name=f"{project_name}-{environment}-api",
            description=f"API Gateway for {project_name}",
            deploy_options=apigw.StageOptions(
                stage_name=environment,
                throttling_rate_limit=1000,
                throttling_burst_limit=2000,
                logging_level=apigw.MethodLoggingLevel.INFO,
                metrics_enabled=True,
                tracing_enabled=True,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=["Content-Type", "Authorization", "X-Api-Key"],
            ),
        )
        
        # API Key and Usage Plan
        api_key = self.api_gateway.add_api_key(
            "ApiKey",
            api_key_name=f"{project_name}-{environment}-key",
        )
        
        usage_plan = self.api_gateway.add_usage_plan(
            "UsagePlan",
            name=f"{project_name}-{environment}-usage-plan",
            throttle=apigw.ThrottleSettings(
                rate_limit=100,
                burst_limit=200,
            ),
            quota=apigw.QuotaSettings(
                limit=10000,
                period=apigw.Period.DAY,
            ),
        )
        
        usage_plan.add_api_key(api_key)
        usage_plan.add_api_stage(
            stage=self.api_gateway.deployment_stage,
        )
        
        # ======================================================================
        # Lambda Functions
        # ======================================================================
        
        # Bedrock Invocation Lambda
        bedrock_lambda = _lambda.Function(
            self,
            "BedrockInvokeLambda",
            function_name=f"{project_name}-{environment}-bedrock-invoke",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_inline("""
import json
import boto3
import os

bedrock_runtime = boto3.client('bedrock-runtime')
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')

def handler(event, context):
    \"\"\"
    Lambda handler for Bedrock model invocation.
    \"\"\"
    try:
        body = json.loads(event.get('body', '{}'))
        prompt = body.get('prompt', '')
        model_id = body.get('model_id', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
        
        # Invoke Bedrock model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'response': response_body.get('content', [{}])[0].get('text', ''),
                'model': model_id,
                'usage': response_body.get('usage', {})
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
"""),
            timeout=cdk.Duration.seconds(120),
            memory_size=512,
            environment={
                "ENVIRONMENT": environment,
                "BEDROCK_AGENT_ID": bedrock_agent_id,
            },
            log_retention=logs.RetentionDays.ONE_WEEK,
        )
        
        # Grant Bedrock permissions
        bedrock_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock-agent-runtime:InvokeAgent",
                ],
                resources=["*"],
            )
        )
        
        # API Gateway integration for Lambda
        bedrock_integration = apigw.LambdaIntegration(
            bedrock_lambda,
            proxy=True,
        )
        
        bedrock_resource = self.api_gateway.root.add_resource("invoke")
        bedrock_resource.add_method(
            "POST",
            bedrock_integration,
            api_key_required=True,
        )
        
        # Agent Invocation Lambda
        agent_lambda = _lambda.Function(
            self,
            "AgentInvokeLambda",
            function_name=f"{project_name}-{environment}-agent-invoke",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_inline("""
import json
import boto3
import os

bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')

def handler(event, context):
    \"\"\"
    Lambda handler for Bedrock Agent invocation.
    \"\"\"
    try:
        body = json.loads(event.get('body', '{}'))
        input_text = body.get('input', '')
        session_id = body.get('session_id', context.aws_request_id)
        agent_id = os.environ.get('BEDROCK_AGENT_ID')
        agent_alias_id = os.environ.get('BEDROCK_AGENT_ALIAS_ID', 'TSTALIASID')
        
        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=input_text,
        )
        
        # Collect response chunks
        output_text = ''
        for event in response.get('completion', []):
            if 'chunk' in event:
                chunk = event['chunk']
                if 'bytes' in chunk:
                    output_text += chunk['bytes'].decode('utf-8')
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'response': output_text,
                'session_id': session_id
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
"""),
            timeout=cdk.Duration.seconds(120),
            memory_size=512,
            environment={
                "ENVIRONMENT": environment,
                "BEDROCK_AGENT_ID": bedrock_agent_id,
            },
            log_retention=logs.RetentionDays.ONE_WEEK,
        )
        
        agent_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock-agent-runtime:InvokeAgent",
                ],
                resources=["*"],
            )
        )
        
        agent_integration = apigw.LambdaIntegration(agent_lambda, proxy=True)
        agent_resource = self.api_gateway.root.add_resource("agent")
        agent_resource.add_method("POST", agent_integration, api_key_required=True)
        
        # ======================================================================
        # Outputs
        # ======================================================================
        CfnOutput(self, "ClusterName", value=self.ecs_cluster.cluster_name)
        CfnOutput(self, "ServiceName", value=self.ecs_service.service.service_name)
        CfnOutput(self, "LoadBalancerDNS", value=self.ecs_service.load_balancer.load_balancer_dns_name)
        CfnOutput(self, "ApiGatewayUrl", value=self.api_gateway.url)
        CfnOutput(self, "ApiKeyId", value=api_key.key_id)

