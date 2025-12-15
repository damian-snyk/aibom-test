"""
Bedrock Stack - Bedrock Agents, Knowledge Bases, AgentCore
"""
from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    aws_iam as iam,
    aws_s3 as s3,
    aws_bedrock as bedrock,
    aws_lambda as _lambda,
    aws_logs as logs,
    CfnOutput,
)


class BedrockStack(cdk.Stack):
    """Bedrock and AgentCore infrastructure stack."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        environment: str,
        project_name: str,
        execution_role: iam.IRole,
        data_bucket: s3.IBucket,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.environment = environment
        self.project_name = project_name
        
        # ======================================================================
        # Bedrock Knowledge Base
        # ======================================================================
        # Note: Knowledge Base requires OpenSearch Serverless collection
        # This is a placeholder - actual KB creation requires additional resources
        
        # Create knowledge base IAM role
        kb_role = iam.Role(
            self,
            "KnowledgeBaseRole",
            role_name=f"{project_name}-{environment}-kb-role",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
        )
        
        kb_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3Access",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:ListBucket",
                ],
                resources=[
                    data_bucket.bucket_arn,
                    f"{data_bucket.bucket_arn}/*",
                ],
            )
        )
        
        kb_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockModelAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                ],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0",
                    "arn:aws:bedrock:*::foundation-model/cohere.embed-english-v3",
                ],
            )
        )
        
        # ======================================================================
        # Bedrock Agent
        # ======================================================================
        self.agent = bedrock.CfnAgent(
            self,
            "BedrockAgent",
            agent_name=f"{project_name}-{environment}-agent",
            description=f"AI Agent for {project_name} application",
            agent_resource_role_arn=execution_role.role_arn,
            foundation_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            idle_session_ttl_in_seconds=600,
            instruction="""You are an intelligent AI assistant for the AIBOM application.
            
Your primary responsibilities are:
1. Answer questions about AI/ML models, datasets, and libraries
2. Help users understand their AI Bill of Materials (AIBOM)
3. Provide guidance on AI security and compliance
4. Assist with code analysis for AI components

Always be helpful, accurate, and security-conscious in your responses.
When discussing AI models, include relevant information about:
- Model capabilities and limitations
- License information
- Potential security considerations
- Best practices for deployment

If you're unsure about something, say so rather than making up information.""",
            auto_prepare=True,
            skip_resource_in_use_check_on_delete=True,
        )
        
        self.agent_id = self.agent.attr_agent_id
        
        # ======================================================================
        # Bedrock Agent Alias
        # ======================================================================
        self.agent_alias = bedrock.CfnAgentAlias(
            self,
            "BedrockAgentAlias",
            agent_id=self.agent.attr_agent_id,
            agent_alias_name=environment,
            description=f"{environment} alias for {project_name} agent",
        )
        
        # ======================================================================
        # Agent Action Group Lambda
        # ======================================================================
        action_group_lambda = _lambda.Function(
            self,
            "AgentActionGroupLambda",
            function_name=f"{project_name}-{environment}-agent-actions",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_inline("""
import json
import boto3

def handler(event, context):
    \"\"\"
    Handle Bedrock Agent action group requests.
    This Lambda processes tool invocations from the Bedrock Agent.
    \"\"\"
    print(f"Received event: {json.dumps(event)}")
    
    action_group = event.get('actionGroup', '')
    api_path = event.get('apiPath', '')
    http_method = event.get('httpMethod', '')
    parameters = event.get('parameters', [])
    request_body = event.get('requestBody', {})
    
    # Parse parameters into a dict
    params = {p['name']: p['value'] for p in parameters}
    
    response_body = {
        "application/json": {
            "body": json.dumps({
                "message": "Action processed successfully",
                "action": api_path,
                "parameters": params
            })
        }
    }
    
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "apiPath": api_path,
            "httpMethod": http_method,
            "httpStatusCode": 200,
            "responseBody": response_body
        }
    }
"""),
            timeout=cdk.Duration.seconds(30),
            memory_size=256,
            log_retention=logs.RetentionDays.ONE_WEEK,
        )
        
        # Grant Bedrock permission to invoke Lambda
        action_group_lambda.add_permission(
            "BedrockInvokePermission",
            principal=iam.ServicePrincipal("bedrock.amazonaws.com"),
            action="lambda:InvokeFunction",
            source_arn=f"arn:aws:bedrock:{self.region}:{self.account}:agent/*",
        )
        
        # ======================================================================
        # Bedrock Guardrail
        # ======================================================================
        self.guardrail = bedrock.CfnGuardrail(
            self,
            "BedrockGuardrail",
            name=f"{project_name}-{environment}-guardrail",
            description="Content filtering guardrail for AI responses",
            blocked_input_messaging="I'm unable to process this request due to content policy restrictions.",
            blocked_outputs_messaging="I'm unable to provide this response due to content policy restrictions.",
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=[
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="HATE",
                        input_strength="HIGH",
                        output_strength="HIGH"
                    ),
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="VIOLENCE",
                        input_strength="HIGH",
                        output_strength="HIGH"
                    ),
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="SEXUAL",
                        input_strength="HIGH",
                        output_strength="HIGH"
                    ),
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="INSULTS",
                        input_strength="MEDIUM",
                        output_strength="MEDIUM"
                    ),
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="MISCONDUCT",
                        input_strength="HIGH",
                        output_strength="HIGH"
                    ),
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="PROMPT_ATTACK",
                        input_strength="HIGH",
                        output_strength="NONE"
                    ),
                ]
            ),
            sensitive_information_policy_config=bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                pii_entities_config=[
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(
                        type="EMAIL",
                        action="ANONYMIZE"
                    ),
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(
                        type="PHONE",
                        action="ANONYMIZE"
                    ),
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(
                        type="US_SOCIAL_SECURITY_NUMBER",
                        action="BLOCK"
                    ),
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(
                        type="CREDIT_DEBIT_CARD_NUMBER",
                        action="BLOCK"
                    ),
                ]
            ),
        )
        
        # ======================================================================
        # Guardrail Version
        # ======================================================================
        guardrail_version = bedrock.CfnGuardrailVersion(
            self,
            "GuardrailVersion",
            guardrail_identifier=self.guardrail.attr_guardrail_id,
            description=f"Version for {environment}",
        )
        
        # ======================================================================
        # Outputs
        # ======================================================================
        CfnOutput(self, "AgentId", value=self.agent.attr_agent_id, export_name=f"{project_name}-{environment}-agent-id")
        CfnOutput(self, "AgentArn", value=self.agent.attr_agent_arn)
        CfnOutput(self, "AgentAliasId", value=self.agent_alias.attr_agent_alias_id)
        CfnOutput(self, "GuardrailId", value=self.guardrail.attr_guardrail_id)
        CfnOutput(self, "ActionGroupLambdaArn", value=action_group_lambda.function_arn)

