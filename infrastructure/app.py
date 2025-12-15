#!/usr/bin/env python3
"""
AWS CDK Application Entry Point
Deploys AI/ML application infrastructure with Bedrock and AgentCore support.
"""
import os
import aws_cdk as cdk
from stacks.network_stack import NetworkStack
from stacks.security_stack import SecurityStack
from stacks.storage_stack import StorageStack
from stacks.compute_stack import ComputeStack
from stacks.bedrock_stack import BedrockStack
from stacks.monitoring_stack import MonitoringStack


def main():
    app = cdk.App()
    
    # Get context values
    environment = app.node.try_get_context("environment") or "staging"
    project_name = app.node.try_get_context("projectName") or "aibom-app"
    
    # Environment configuration
    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1")
    )
    
    # Common tags
    common_tags = {
        "Project": project_name,
        "Environment": environment,
        "ManagedBy": "CDK",
        "Application": "AIBOM-Test"
    }
    
    # Stack naming convention
    stack_prefix = f"{project_name}-{environment}"
    
    # ==========================================================================
    # Network Stack - VPC, Subnets, Security Groups
    # ==========================================================================
    network_stack = NetworkStack(
        app,
        f"{stack_prefix}-network",
        env=env,
        environment=environment,
        project_name=project_name,
        description=f"Network infrastructure for {project_name}"
    )
    cdk.Tags.of(network_stack).add("Stack", "Network")
    
    # ==========================================================================
    # Security Stack - IAM Roles, Policies, Secrets
    # ==========================================================================
    security_stack = SecurityStack(
        app,
        f"{stack_prefix}-security",
        env=env,
        environment=environment,
        project_name=project_name,
        description=f"Security infrastructure for {project_name}"
    )
    cdk.Tags.of(security_stack).add("Stack", "Security")
    
    # ==========================================================================
    # Storage Stack - S3, DynamoDB, ElastiCache
    # ==========================================================================
    storage_stack = StorageStack(
        app,
        f"{stack_prefix}-storage",
        env=env,
        environment=environment,
        project_name=project_name,
        vpc=network_stack.vpc,
        description=f"Storage infrastructure for {project_name}"
    )
    storage_stack.add_dependency(network_stack)
    cdk.Tags.of(storage_stack).add("Stack", "Storage")
    
    # ==========================================================================
    # Bedrock Stack - Bedrock Agents, Knowledge Bases, AgentCore
    # ==========================================================================
    bedrock_stack = BedrockStack(
        app,
        f"{stack_prefix}-bedrock",
        env=env,
        environment=environment,
        project_name=project_name,
        execution_role=security_stack.bedrock_execution_role,
        data_bucket=storage_stack.data_bucket,
        description=f"Bedrock and AgentCore infrastructure for {project_name}"
    )
    bedrock_stack.add_dependency(security_stack)
    bedrock_stack.add_dependency(storage_stack)
    cdk.Tags.of(bedrock_stack).add("Stack", "Bedrock")
    
    # ==========================================================================
    # Compute Stack - ECS, Lambda, API Gateway
    # ==========================================================================
    compute_stack = ComputeStack(
        app,
        f"{stack_prefix}-compute",
        env=env,
        environment=environment,
        project_name=project_name,
        vpc=network_stack.vpc,
        execution_role=security_stack.ecs_execution_role,
        task_role=security_stack.ecs_task_role,
        data_bucket=storage_stack.data_bucket,
        bedrock_agent_id=bedrock_stack.agent_id,
        description=f"Compute infrastructure for {project_name}"
    )
    compute_stack.add_dependency(network_stack)
    compute_stack.add_dependency(security_stack)
    compute_stack.add_dependency(storage_stack)
    compute_stack.add_dependency(bedrock_stack)
    cdk.Tags.of(compute_stack).add("Stack", "Compute")
    
    # ==========================================================================
    # Monitoring Stack - CloudWatch, X-Ray, Alarms
    # ==========================================================================
    monitoring_stack = MonitoringStack(
        app,
        f"{stack_prefix}-monitoring",
        env=env,
        environment=environment,
        project_name=project_name,
        ecs_cluster=compute_stack.ecs_cluster,
        ecs_service=compute_stack.ecs_service,
        api_gateway=compute_stack.api_gateway,
        description=f"Monitoring infrastructure for {project_name}"
    )
    monitoring_stack.add_dependency(compute_stack)
    cdk.Tags.of(monitoring_stack).add("Stack", "Monitoring")
    
    # Apply common tags to all stacks
    for key, value in common_tags.items():
        cdk.Tags.of(app).add(key, value)
    
    app.synth()


if __name__ == "__main__":
    main()

