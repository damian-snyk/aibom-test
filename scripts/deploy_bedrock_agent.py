#!/usr/bin/env python3
"""
Bedrock Agent Deployment Script

Deploys and configures Bedrock Agents and Knowledge Bases based on YAML configuration.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import boto3
import yaml


class BedrockAgentDeployer:
    """Deploy and manage Bedrock Agents."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.bedrock_agent = boto3.client("bedrock-agent", region_name=region)
        self.bedrock = boto3.client("bedrock", region_name=region)
        self.iam = boto3.client("iam")
        self.s3 = boto3.client("s3")
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def create_or_update_agent(self, config: dict) -> str:
        """Create or update a Bedrock Agent."""
        agent_config = config.get("agent", {})
        agent_name = agent_config.get("name")
        
        # Check if agent exists
        existing_agent = self._find_agent_by_name(agent_name)
        
        if existing_agent:
            print(f"Updating existing agent: {agent_name}")
            response = self.bedrock_agent.update_agent(
                agentId=existing_agent["agentId"],
                agentName=agent_name,
                instruction=agent_config.get("instruction", ""),
                foundationModel=agent_config.get("foundation_model"),
                idleSessionTTLInSeconds=agent_config.get("idle_session_ttl", 600),
                agentResourceRoleArn=agent_config.get("execution_role_arn"),
            )
            agent_id = existing_agent["agentId"]
        else:
            print(f"Creating new agent: {agent_name}")
            response = self.bedrock_agent.create_agent(
                agentName=agent_name,
                instruction=agent_config.get("instruction", ""),
                foundationModel=agent_config.get("foundation_model"),
                idleSessionTTLInSeconds=agent_config.get("idle_session_ttl", 600),
                agentResourceRoleArn=agent_config.get("execution_role_arn"),
            )
            agent_id = response["agent"]["agentId"]
        
        print(f"Agent ID: {agent_id}")
        
        # Wait for agent to be ready
        self._wait_for_agent_status(agent_id, "NOT_PREPARED")
        
        return agent_id
    
    def create_action_groups(self, agent_id: str, config: dict) -> None:
        """Create action groups for the agent."""
        action_groups = config.get("action_groups", [])
        
        for ag in action_groups:
            print(f"Creating action group: {ag['name']}")
            
            try:
                self.bedrock_agent.create_agent_action_group(
                    agentId=agent_id,
                    agentVersion="DRAFT",
                    actionGroupName=ag["name"],
                    description=ag.get("description", ""),
                    actionGroupExecutor={
                        "lambda": ag["lambda_arn"]
                    },
                    apiSchema={
                        "payload": ag.get("api_schema", "")
                    } if ag.get("api_schema") else {
                        "s3": {
                            "s3BucketName": ag.get("schema_bucket"),
                            "s3ObjectKey": ag.get("schema_key"),
                        }
                    },
                )
            except self.bedrock_agent.exceptions.ConflictException:
                print(f"Action group {ag['name']} already exists, updating...")
                self.bedrock_agent.update_agent_action_group(
                    agentId=agent_id,
                    agentVersion="DRAFT",
                    actionGroupId=self._get_action_group_id(agent_id, ag["name"]),
                    actionGroupName=ag["name"],
                    description=ag.get("description", ""),
                    actionGroupExecutor={
                        "lambda": ag["lambda_arn"]
                    },
                )
    
    def create_knowledge_base(self, config: dict) -> str | None:
        """Create a Knowledge Base for the agent."""
        kb_config = config.get("knowledge_base")
        if not kb_config:
            return None
        
        print(f"Creating knowledge base: {kb_config['name']}")
        
        # Check if KB exists
        existing_kb = self._find_knowledge_base_by_name(kb_config["name"])
        
        if existing_kb:
            print(f"Knowledge base already exists: {existing_kb['knowledgeBaseId']}")
            return existing_kb["knowledgeBaseId"]
        
        response = self.bedrock_agent.create_knowledge_base(
            name=kb_config["name"],
            description=kb_config.get("description", ""),
            roleArn=kb_config["role_arn"],
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{self.region}::foundation-model/{kb_config.get('embedding_model', 'amazon.titan-embed-text-v2:0')}"
                }
            },
            storageConfiguration=kb_config.get("storage_configuration", {
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": kb_config.get("collection_arn", ""),
                    "vectorIndexName": kb_config.get("index_name", "bedrock-kb-index"),
                    "fieldMapping": {
                        "vectorField": "vector",
                        "textField": "text",
                        "metadataField": "metadata"
                    }
                }
            }),
        )
        
        kb_id = response["knowledgeBase"]["knowledgeBaseId"]
        print(f"Knowledge Base ID: {kb_id}")
        
        # Create data source
        if kb_config.get("data_source"):
            ds_config = kb_config["data_source"]
            print(f"Creating data source: {ds_config['name']}")
            
            self.bedrock_agent.create_data_source(
                knowledgeBaseId=kb_id,
                name=ds_config["name"],
                description=ds_config.get("description", ""),
                dataSourceConfiguration={
                    "type": "S3",
                    "s3Configuration": {
                        "bucketArn": ds_config["bucket_arn"],
                        "inclusionPrefixes": ds_config.get("prefixes", [])
                    }
                },
            )
        
        return kb_id
    
    def associate_knowledge_base(self, agent_id: str, kb_id: str, config: dict) -> None:
        """Associate a Knowledge Base with an Agent."""
        kb_config = config.get("knowledge_base", {})
        
        print(f"Associating knowledge base {kb_id} with agent {agent_id}")
        
        try:
            self.bedrock_agent.associate_agent_knowledge_base(
                agentId=agent_id,
                agentVersion="DRAFT",
                knowledgeBaseId=kb_id,
                description=kb_config.get("description", "Knowledge base for agent"),
            )
        except self.bedrock_agent.exceptions.ConflictException:
            print("Knowledge base already associated")
    
    def prepare_agent(self, agent_id: str) -> None:
        """Prepare the agent for deployment."""
        print(f"Preparing agent: {agent_id}")
        
        self.bedrock_agent.prepare_agent(agentId=agent_id)
        self._wait_for_agent_status(agent_id, "PREPARED")
        
        print("Agent prepared successfully")
    
    def create_agent_alias(self, agent_id: str, alias_name: str) -> str:
        """Create or update an agent alias."""
        print(f"Creating/updating alias: {alias_name}")
        
        # Get agent version
        agent = self.bedrock_agent.get_agent(agentId=agent_id)
        
        # Check for existing alias
        existing_alias = self._find_alias_by_name(agent_id, alias_name)
        
        if existing_alias:
            response = self.bedrock_agent.update_agent_alias(
                agentId=agent_id,
                agentAliasId=existing_alias["agentAliasId"],
                agentAliasName=alias_name,
            )
            alias_id = existing_alias["agentAliasId"]
        else:
            response = self.bedrock_agent.create_agent_alias(
                agentId=agent_id,
                agentAliasName=alias_name,
            )
            alias_id = response["agentAlias"]["agentAliasId"]
        
        print(f"Alias ID: {alias_id}")
        return alias_id
    
    def deploy(self, config_path: str, environment: str) -> dict:
        """Full deployment workflow."""
        config = self.load_config(config_path)
        
        # Create/update agent
        agent_id = self.create_or_update_agent(config)
        
        # Create action groups
        self.create_action_groups(agent_id, config)
        
        # Create knowledge base
        kb_id = self.create_knowledge_base(config)
        if kb_id:
            self.associate_knowledge_base(agent_id, kb_id, config)
        
        # Prepare agent
        self.prepare_agent(agent_id)
        
        # Create alias
        alias_id = self.create_agent_alias(agent_id, environment)
        
        return {
            "agent_id": agent_id,
            "alias_id": alias_id,
            "knowledge_base_id": kb_id,
        }
    
    def _find_agent_by_name(self, name: str) -> dict | None:
        """Find an agent by name."""
        paginator = self.bedrock_agent.get_paginator("list_agents")
        for page in paginator.paginate():
            for agent in page["agentSummaries"]:
                if agent["agentName"] == name:
                    return agent
        return None
    
    def _find_knowledge_base_by_name(self, name: str) -> dict | None:
        """Find a knowledge base by name."""
        paginator = self.bedrock_agent.get_paginator("list_knowledge_bases")
        for page in paginator.paginate():
            for kb in page["knowledgeBaseSummaries"]:
                if kb["name"] == name:
                    return kb
        return None
    
    def _find_alias_by_name(self, agent_id: str, alias_name: str) -> dict | None:
        """Find an alias by name."""
        try:
            response = self.bedrock_agent.list_agent_aliases(agentId=agent_id)
            for alias in response["agentAliasSummaries"]:
                if alias["agentAliasName"] == alias_name:
                    return alias
        except Exception:
            pass
        return None
    
    def _get_action_group_id(self, agent_id: str, name: str) -> str | None:
        """Get action group ID by name."""
        response = self.bedrock_agent.list_agent_action_groups(
            agentId=agent_id,
            agentVersion="DRAFT"
        )
        for ag in response["actionGroupSummaries"]:
            if ag["actionGroupName"] == name:
                return ag["actionGroupId"]
        return None
    
    def _wait_for_agent_status(self, agent_id: str, target_status: str, timeout: int = 300) -> None:
        """Wait for agent to reach target status."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.bedrock_agent.get_agent(agentId=agent_id)
            status = response["agent"]["agentStatus"]
            print(f"Agent status: {status}")
            
            if status == target_status:
                return
            if status in ["FAILED", "DELETING"]:
                raise Exception(f"Agent reached unexpected status: {status}")
            
            time.sleep(5)
        
        raise TimeoutError(f"Agent did not reach status {target_status} within {timeout}s")


def main():
    parser = argparse.ArgumentParser(description="Deploy Bedrock Agent")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument("--environment", "-e", required=True, help="Deployment environment")
    parser.add_argument("--region", "-r", default="us-east-1", help="AWS region")
    
    args = parser.parse_args()
    
    deployer = BedrockAgentDeployer(region=args.region)
    
    try:
        result = deployer.deploy(args.config, args.environment)
        print("\n" + "=" * 50)
        print("Deployment completed successfully!")
        print(json.dumps(result, indent=2))
        print("=" * 50)
    except Exception as e:
        print(f"Deployment failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

