"""
Monitoring Stack - CloudWatch, X-Ray, Alarms, Dashboards
"""
from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subscriptions,
    aws_logs as logs,
    aws_ecs as ecs,
    aws_apigateway as apigw,
    CfnOutput,
)


class MonitoringStack(cdk.Stack):
    """Monitoring infrastructure stack."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        environment: str,
        project_name: str,
        ecs_cluster: ecs.ICluster,
        ecs_service,  # ApplicationLoadBalancedFargateService
        api_gateway: apigw.RestApi,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.environment = environment
        self.project_name = project_name
        is_production = environment == "production"
        
        # ======================================================================
        # SNS Topic for Alerts
        # ======================================================================
        self.alert_topic = sns.Topic(
            self,
            "AlertTopic",
            topic_name=f"{project_name}-{environment}-alerts",
            display_name=f"{project_name} {environment} Alerts",
        )
        
        # Add email subscription placeholder
        # Uncomment and modify with actual email
        # self.alert_topic.add_subscription(
        #     sns_subscriptions.EmailSubscription("alerts@example.com")
        # )
        
        # ======================================================================
        # CloudWatch Log Groups
        # ======================================================================
        self.application_log_group = logs.LogGroup(
            self,
            "ApplicationLogGroup",
            log_group_name=f"/aws/{project_name}/{environment}/application",
            retention=logs.RetentionDays.ONE_MONTH if is_production else logs.RetentionDays.ONE_WEEK,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        
        self.bedrock_log_group = logs.LogGroup(
            self,
            "BedrockLogGroup",
            log_group_name=f"/aws/{project_name}/{environment}/bedrock",
            retention=logs.RetentionDays.ONE_MONTH if is_production else logs.RetentionDays.ONE_WEEK,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        
        # ======================================================================
        # CloudWatch Alarms
        # ======================================================================
        
        # ECS CPU Utilization Alarm
        ecs_cpu_alarm = cloudwatch.Alarm(
            self,
            "ECSCPUAlarm",
            alarm_name=f"{project_name}-{environment}-ecs-cpu-high",
            alarm_description="ECS CPU utilization is too high",
            metric=ecs_service.service.metric_cpu_utilization(),
            threshold=80,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        ecs_cpu_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))
        
        # ECS Memory Utilization Alarm
        ecs_memory_alarm = cloudwatch.Alarm(
            self,
            "ECSMemoryAlarm",
            alarm_name=f"{project_name}-{environment}-ecs-memory-high",
            alarm_description="ECS memory utilization is too high",
            metric=ecs_service.service.metric_memory_utilization(),
            threshold=85,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        ecs_memory_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))
        
        # API Gateway 5XX Errors Alarm
        api_5xx_alarm = cloudwatch.Alarm(
            self,
            "API5XXAlarm",
            alarm_name=f"{project_name}-{environment}-api-5xx-errors",
            alarm_description="API Gateway is returning 5XX errors",
            metric=api_gateway.metric_server_error(
                period=cdk.Duration.minutes(5),
                statistic="Sum",
            ),
            threshold=10,
            evaluation_periods=2,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        api_5xx_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))
        
        # API Gateway 4XX Errors Alarm
        api_4xx_alarm = cloudwatch.Alarm(
            self,
            "API4XXAlarm",
            alarm_name=f"{project_name}-{environment}-api-4xx-errors",
            alarm_description="API Gateway is returning high 4XX errors",
            metric=api_gateway.metric_client_error(
                period=cdk.Duration.minutes(5),
                statistic="Sum",
            ),
            threshold=100,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        api_4xx_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))
        
        # API Gateway Latency Alarm
        api_latency_alarm = cloudwatch.Alarm(
            self,
            "APILatencyAlarm",
            alarm_name=f"{project_name}-{environment}-api-latency-high",
            alarm_description="API Gateway latency is too high",
            metric=api_gateway.metric_latency(
                period=cdk.Duration.minutes(5),
                statistic="p99",
            ),
            threshold=5000,  # 5 seconds
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        api_latency_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))
        
        # ======================================================================
        # CloudWatch Dashboard
        # ======================================================================
        dashboard = cloudwatch.Dashboard(
            self,
            "Dashboard",
            dashboard_name=f"{project_name}-{environment}-dashboard",
        )
        
        # ECS Metrics Row
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown="# ECS Service Metrics",
                width=24,
                height=1,
            ),
        )
        
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="ECS CPU Utilization",
                left=[ecs_service.service.metric_cpu_utilization()],
                width=8,
                height=6,
            ),
            cloudwatch.GraphWidget(
                title="ECS Memory Utilization",
                left=[ecs_service.service.metric_memory_utilization()],
                width=8,
                height=6,
            ),
            cloudwatch.SingleValueWidget(
                title="Running Tasks",
                metrics=[
                    cloudwatch.Metric(
                        namespace="AWS/ECS",
                        metric_name="RunningTaskCount",
                        dimensions_map={
                            "ClusterName": ecs_cluster.cluster_name,
                            "ServiceName": ecs_service.service.service_name,
                        },
                        statistic="Average",
                    )
                ],
                width=8,
                height=6,
            ),
        )
        
        # API Gateway Metrics Row
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown="# API Gateway Metrics",
                width=24,
                height=1,
            ),
        )
        
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="API Requests",
                left=[api_gateway.metric_count(statistic="Sum")],
                width=8,
                height=6,
            ),
            cloudwatch.GraphWidget(
                title="API Latency (p99)",
                left=[api_gateway.metric_latency(statistic="p99")],
                width=8,
                height=6,
            ),
            cloudwatch.GraphWidget(
                title="API Errors",
                left=[
                    api_gateway.metric_client_error(statistic="Sum"),
                    api_gateway.metric_server_error(statistic="Sum"),
                ],
                width=8,
                height=6,
            ),
        )
        
        # Bedrock Metrics Row
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown="# Bedrock Metrics",
                width=24,
                height=1,
            ),
        )
        
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Bedrock Model Invocations",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/Bedrock",
                        metric_name="Invocations",
                        dimensions_map={"ModelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
                        statistic="Sum",
                    ),
                    cloudwatch.Metric(
                        namespace="AWS/Bedrock",
                        metric_name="Invocations",
                        dimensions_map={"ModelId": "amazon.titan-embed-text-v2:0"},
                        statistic="Sum",
                    ),
                ],
                width=12,
                height=6,
            ),
            cloudwatch.GraphWidget(
                title="Bedrock Latency",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/Bedrock",
                        metric_name="InvocationLatency",
                        dimensions_map={"ModelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
                        statistic="p99",
                    ),
                ],
                width=12,
                height=6,
            ),
        )
        
        # Alarms Status Row
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown="# Alarm Status",
                width=24,
                height=1,
            ),
        )
        
        dashboard.add_widgets(
            cloudwatch.AlarmStatusWidget(
                title="All Alarms",
                alarms=[
                    ecs_cpu_alarm,
                    ecs_memory_alarm,
                    api_5xx_alarm,
                    api_4xx_alarm,
                    api_latency_alarm,
                ],
                width=24,
                height=4,
            ),
        )
        
        # ======================================================================
        # Outputs
        # ======================================================================
        CfnOutput(self, "AlertTopicArn", value=self.alert_topic.topic_arn)
        CfnOutput(self, "DashboardName", value=dashboard.dashboard_name)
        CfnOutput(
            self,
            "DashboardUrl",
            value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={dashboard.dashboard_name}"
        )

