"""
Storage Stack - S3, DynamoDB, ElastiCache
"""
from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    aws_elasticache as elasticache,
    aws_ec2 as ec2,
    CfnOutput,
    RemovalPolicy,
)


class StorageStack(cdk.Stack):
    """Storage infrastructure stack."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        environment: str,
        project_name: str,
        vpc: ec2.IVpc,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.environment = environment
        self.project_name = project_name
        is_production = environment == "production"
        
        # ======================================================================
        # S3 Bucket - Application Data
        # ======================================================================
        self.data_bucket = s3.Bucket(
            self,
            "DataBucket",
            bucket_name=f"{project_name}-{environment}-data-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN if is_production else RemovalPolicy.DESTROY,
            auto_delete_objects=not is_production,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="TransitionToIA",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=cdk.Duration.days(30),
                        ),
                    ],
                ),
                s3.LifecycleRule(
                    id="ExpireOldVersions",
                    enabled=True,
                    noncurrent_version_expiration=cdk.Duration.days(90),
                ),
            ],
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT, s3.HttpMethods.POST],
                    allowed_origins=["*"],
                    allowed_headers=["*"],
                    max_age=3000,
                )
            ],
        )
        
        # ======================================================================
        # S3 Bucket - Knowledge Base Documents
        # ======================================================================
        self.knowledge_base_bucket = s3.Bucket(
            self,
            "KnowledgeBaseBucket",
            bucket_name=f"{project_name}-{environment}-knowledge-base-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN if is_production else RemovalPolicy.DESTROY,
            auto_delete_objects=not is_production,
        )
        
        # ======================================================================
        # S3 Bucket - Model Artifacts
        # ======================================================================
        self.model_artifacts_bucket = s3.Bucket(
            self,
            "ModelArtifactsBucket",
            bucket_name=f"{project_name}-{environment}-model-artifacts-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN if is_production else RemovalPolicy.DESTROY,
            auto_delete_objects=not is_production,
        )
        
        # ======================================================================
        # DynamoDB Table - Conversations
        # ======================================================================
        self.conversations_table = dynamodb.Table(
            self,
            "ConversationsTable",
            table_name=f"{project_name}-{environment}-conversations",
            partition_key=dynamodb.Attribute(
                name="conversation_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN if is_production else RemovalPolicy.DESTROY,
            point_in_time_recovery=is_production,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            time_to_live_attribute="ttl",
        )
        
        # GSI for user queries
        self.conversations_table.add_global_secondary_index(
            index_name="user-index",
            partition_key=dynamodb.Attribute(
                name="user_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )
        
        # ======================================================================
        # DynamoDB Table - Embeddings Cache
        # ======================================================================
        self.embeddings_table = dynamodb.Table(
            self,
            "EmbeddingsTable",
            table_name=f"{project_name}-{environment}-embeddings",
            partition_key=dynamodb.Attribute(
                name="text_hash",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            time_to_live_attribute="ttl",
        )
        
        # ======================================================================
        # DynamoDB Table - Agent Sessions
        # ======================================================================
        self.agent_sessions_table = dynamodb.Table(
            self,
            "AgentSessionsTable",
            table_name=f"{project_name}-{environment}-agent-sessions",
            partition_key=dynamodb.Attribute(
                name="session_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="agent_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            time_to_live_attribute="ttl",
        )
        
        # ======================================================================
        # ElastiCache Redis Cluster
        # ======================================================================
        cache_subnet_group = elasticache.CfnSubnetGroup(
            self,
            "CacheSubnetGroup",
            subnet_group_name=f"{project_name}-{environment}-cache-subnets",
            description="Subnet group for ElastiCache",
            subnet_ids=[subnet.subnet_id for subnet in vpc.private_subnets],
        )
        
        cache_security_group = ec2.SecurityGroup(
            self,
            "CacheSecurityGroup",
            vpc=vpc,
            security_group_name=f"{project_name}-{environment}-cache-sg",
            description="Security group for ElastiCache Redis",
            allow_all_outbound=False,
        )
        
        # Allow access from private subnets
        cache_security_group.add_ingress_rule(
            ec2.Peer.ipv4(vpc.vpc_cidr_block),
            ec2.Port.tcp(6379),
            "Allow Redis from VPC"
        )
        
        self.redis_cluster = elasticache.CfnReplicationGroup(
            self,
            "RedisCluster",
            replication_group_id=f"{project_name}-{environment}-redis",
            replication_group_description=f"Redis cluster for {project_name}",
            engine="redis",
            engine_version="7.1",
            cache_node_type="cache.t4g.micro" if not is_production else "cache.r7g.large",
            num_cache_clusters=2 if is_production else 1,
            automatic_failover_enabled=is_production,
            multi_az_enabled=is_production,
            cache_subnet_group_name=cache_subnet_group.subnet_group_name,
            security_group_ids=[cache_security_group.security_group_id],
            at_rest_encryption_enabled=True,
            transit_encryption_enabled=True,
            snapshot_retention_limit=7 if is_production else 1,
            snapshot_window="03:00-05:00",
            preferred_maintenance_window="sun:05:00-sun:07:00",
        )
        
        self.redis_cluster.add_dependency(cache_subnet_group)
        
        # ======================================================================
        # Outputs
        # ======================================================================
        CfnOutput(self, "DataBucketName", value=self.data_bucket.bucket_name)
        CfnOutput(self, "DataBucketArn", value=self.data_bucket.bucket_arn)
        CfnOutput(self, "KnowledgeBaseBucketName", value=self.knowledge_base_bucket.bucket_name)
        CfnOutput(self, "ConversationsTableName", value=self.conversations_table.table_name)
        CfnOutput(self, "EmbeddingsTableName", value=self.embeddings_table.table_name)
        CfnOutput(self, "RedisEndpoint", value=self.redis_cluster.attr_primary_end_point_address)

