"""
Network Stack - VPC, Subnets, Security Groups, VPC Endpoints
"""
from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    CfnOutput,
)


class NetworkStack(cdk.Stack):
    """Network infrastructure stack."""
    
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
        # VPC
        # ======================================================================
        self.vpc = ec2.Vpc(
            self,
            "VPC",
            vpc_name=f"{project_name}-{environment}-vpc",
            max_azs=3,
            nat_gateways=2 if environment == "production" else 1,
            ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24
                ),
            ],
            enable_dns_hostnames=True,
            enable_dns_support=True,
        )
        
        # ======================================================================
        # VPC Flow Logs
        # ======================================================================
        self.vpc.add_flow_log(
            "FlowLog",
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(),
            traffic_type=ec2.FlowLogTrafficType.ALL,
        )
        
        # ======================================================================
        # Security Groups
        # ======================================================================
        
        # ALB Security Group
        self.alb_security_group = ec2.SecurityGroup(
            self,
            "ALBSecurityGroup",
            vpc=self.vpc,
            security_group_name=f"{project_name}-{environment}-alb-sg",
            description="Security group for Application Load Balancer",
            allow_all_outbound=True,
        )
        self.alb_security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(443),
            "Allow HTTPS from anywhere"
        )
        self.alb_security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(80),
            "Allow HTTP from anywhere (redirect to HTTPS)"
        )
        
        # ECS Security Group
        self.ecs_security_group = ec2.SecurityGroup(
            self,
            "ECSSecurityGroup",
            vpc=self.vpc,
            security_group_name=f"{project_name}-{environment}-ecs-sg",
            description="Security group for ECS tasks",
            allow_all_outbound=True,
        )
        self.ecs_security_group.add_ingress_rule(
            self.alb_security_group,
            ec2.Port.tcp(8080),
            "Allow traffic from ALB"
        )
        
        # Lambda Security Group
        self.lambda_security_group = ec2.SecurityGroup(
            self,
            "LambdaSecurityGroup",
            vpc=self.vpc,
            security_group_name=f"{project_name}-{environment}-lambda-sg",
            description="Security group for Lambda functions",
            allow_all_outbound=True,
        )
        
        # Database Security Group
        self.database_security_group = ec2.SecurityGroup(
            self,
            "DatabaseSecurityGroup",
            vpc=self.vpc,
            security_group_name=f"{project_name}-{environment}-db-sg",
            description="Security group for database instances",
            allow_all_outbound=False,
        )
        self.database_security_group.add_ingress_rule(
            self.ecs_security_group,
            ec2.Port.tcp(5432),
            "Allow PostgreSQL from ECS"
        )
        self.database_security_group.add_ingress_rule(
            self.lambda_security_group,
            ec2.Port.tcp(5432),
            "Allow PostgreSQL from Lambda"
        )
        
        # ======================================================================
        # VPC Endpoints for AWS Services
        # ======================================================================
        
        # S3 Gateway Endpoint (free)
        self.vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )
        
        # DynamoDB Gateway Endpoint (free)
        self.vpc.add_gateway_endpoint(
            "DynamoDBEndpoint",
            service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
        )
        
        # Interface endpoints for Bedrock and other services
        endpoint_services = [
            ("ECRApi", ec2.InterfaceVpcEndpointAwsService.ECR),
            ("ECRDocker", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER),
            ("CloudWatchLogs", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
            ("SecretsManager", ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER),
            ("SSM", ec2.InterfaceVpcEndpointAwsService.SSM),
            ("Bedrock", ec2.InterfaceVpcEndpointAwsService.BEDROCK),
            ("BedrockRuntime", ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME),
            ("BedrockAgent", ec2.InterfaceVpcEndpointAwsService.BEDROCK_AGENT),
            ("BedrockAgentRuntime", ec2.InterfaceVpcEndpointAwsService.BEDROCK_AGENT_RUNTIME),
        ]
        
        for endpoint_name, service in endpoint_services:
            self.vpc.add_interface_endpoint(
                endpoint_name,
                service=service,
                private_dns_enabled=True,
                subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            )
        
        # ======================================================================
        # Outputs
        # ======================================================================
        CfnOutput(self, "VpcId", value=self.vpc.vpc_id, export_name=f"{project_name}-{environment}-vpc-id")
        CfnOutput(self, "VpcCidr", value=self.vpc.vpc_cidr_block)
        CfnOutput(self, "ALBSecurityGroupId", value=self.alb_security_group.security_group_id)
        CfnOutput(self, "ECSSecurityGroupId", value=self.ecs_security_group.security_group_id)

