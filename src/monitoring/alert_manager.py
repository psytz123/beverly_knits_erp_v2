"""
Alert Manager
Complete alert system with email, SMS, webhook, and Slack support
"""

import smtplib
import json
import time
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
from queue import Queue, PriorityQueue

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    category: str
    data: Dict[str, Any]
    created_at: datetime
    channels: List[AlertChannel]
    metadata: Dict[str, Any] = None
    
    def __lt__(self, other):
        """Compare alerts by severity for priority queue"""
        return self.severity.value < other.severity.value


class AlertRule:
    """Alert rule definition"""
    
    def __init__(self,
                 name: str,
                 condition: Callable[[Dict], bool],
                 severity: AlertSeverity,
                 channels: List[AlertChannel],
                 message_template: str,
                 cooldown_minutes: int = 30):
        """
        Initialize alert rule
        
        Args:
            name: Rule name
            condition: Function to evaluate condition
            severity: Alert severity
            channels: Delivery channels
            message_template: Message template with {placeholders}
            cooldown_minutes: Minutes before rule can fire again
        """
        self.name = name
        self.condition = condition
        self.severity = severity
        self.channels = channels
        self.message_template = message_template
        self.cooldown_minutes = cooldown_minutes
        self.last_fired = None
    
    def evaluate(self, data: Dict[str, Any]) -> Optional[Alert]:
        """
        Evaluate rule against data
        
        Args:
            data: Data to evaluate
            
        Returns:
            Alert if condition met, None otherwise
        """
        # Check cooldown
        if self.last_fired:
            cooldown_expires = self.last_fired + timedelta(minutes=self.cooldown_minutes)
            if datetime.now() < cooldown_expires:
                return None
        
        # Evaluate condition
        try:
            if self.condition(data):
                # Generate alert
                alert = Alert(
                    id=f"{self.name}_{int(time.time())}",
                    title=f"Alert: {self.name}",
                    message=self.message_template.format(**data),
                    severity=self.severity,
                    category=self.name,
                    data=data,
                    created_at=datetime.now(),
                    channels=self.channels
                )
                
                self.last_fired = datetime.now()
                return alert
        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {e}")
        
        return None


class AlertManager:
    """Complete alert system with multiple channels"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager
        
        Args:
            config: Configuration with channel settings
        """
        self.config = config
        self.smtp_config = config.get('smtp', {})
        self.webhook_urls = config.get('webhooks', {})
        self.sms_config = config.get('sms', {})
        self.slack_config = config.get('slack', {})
        self.teams_config = config.get('teams', {})
        
        # Alert management
        self.rules = {}
        self.alert_queue = PriorityQueue()
        self.alert_history = []
        self.max_history = 1000
        
        # Delivery statistics
        self.delivery_stats = {
            'sent': 0,
            'failed': 0,
            'by_channel': {},
            'by_severity': {}
        }
        
        # Start alert processor thread
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.processor_thread.start()
        
        # Load default rules
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default alert rules"""
        
        # Yarn shortage critical
        self.add_rule(AlertRule(
            name='yarn_shortage_critical',
            condition=lambda data: data.get('shortage_count', 0) > 10,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            message_template='CRITICAL: {shortage_count} yarns with negative balance. Immediate action required.'
        ))
        
        # Capacity overload
        self.add_rule(AlertRule(
            name='capacity_overload',
            condition=lambda data: data.get('utilization', 0) > 95,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            message_template='HIGH: Production capacity at {utilization}%. Risk of delays.'
        ))
        
        # Forecast deviation
        self.add_rule(AlertRule(
            name='forecast_deviation',
            condition=lambda data: abs(data.get('deviation', 0)) > 20,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.EMAIL],
            message_template='Forecast deviation of {deviation}% detected. Review planning.'
        ))
        
        # Order delays
        self.add_rule(AlertRule(
            name='order_delays',
            condition=lambda data: data.get('delayed_orders', 0) > 5,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            message_template='{delayed_orders} production orders are delayed.'
        ))
        
        # Low inventory
        self.add_rule(AlertRule(
            name='low_inventory',
            condition=lambda data: data.get('low_stock_items', 0) > 20,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.EMAIL],
            message_template='{low_stock_items} items below safety stock level.'
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, name: str):
        """Remove alert rule"""
        if name in self.rules:
            del self.rules[name]
            logger.info(f"Removed alert rule: {name}")
    
    def check_and_alert(self, metric_name: str, data: Dict[str, Any]):
        """
        Check metrics against rules and send alerts
        
        Args:
            metric_name: Name of metric
            data: Metric data
        """
        # Check specific rule if exists
        if metric_name in self.rules:
            alert = self.rules[metric_name].evaluate(data)
            if alert:
                self.queue_alert(alert)
        
        # Check all rules
        for rule in self.rules.values():
            alert = rule.evaluate(data)
            if alert:
                self.queue_alert(alert)
    
    def queue_alert(self, alert: Alert):
        """Queue alert for delivery"""
        self.alert_queue.put((alert.severity.value, alert))
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
    
    def send_alert(self, alert: Alert):
        """
        Send alert through specified channels
        
        Args:
            alert: Alert to send
        """
        success_channels = []
        failed_channels = []
        
        for channel in alert.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    self._send_email(alert)
                    success_channels.append(channel)
                    
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook(alert)
                    success_channels.append(channel)
                    
                elif channel == AlertChannel.SLACK:
                    self._send_slack(alert)
                    success_channels.append(channel)
                    
                elif channel == AlertChannel.TEAMS:
                    self._send_teams(alert)
                    success_channels.append(channel)
                    
                elif channel == AlertChannel.SMS:
                    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                        self._send_sms(alert)
                        success_channels.append(channel)
                    
                elif channel == AlertChannel.DISCORD:
                    self._send_discord(alert)
                    success_channels.append(channel)
                    
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
                failed_channels.append(channel)
        
        # Update statistics
        if success_channels:
            self.delivery_stats['sent'] += len(success_channels)
            for channel in success_channels:
                self.delivery_stats['by_channel'][channel.value] = \
                    self.delivery_stats['by_channel'].get(channel.value, 0) + 1
        
        if failed_channels:
            self.delivery_stats['failed'] += len(failed_channels)
        
        self.delivery_stats['by_severity'][alert.severity.name] = \
            self.delivery_stats['by_severity'].get(alert.severity.name, 0) + 1
    
    def _process_alerts(self):
        """Background thread to process alert queue"""
        while self.running:
            try:
                # Get alert from queue (blocks for up to 1 second)
                priority, alert = self.alert_queue.get(timeout=1)
                
                # Send alert
                self.send_alert(alert)
                
            except:
                # Queue empty or timeout
                continue
    
    def _send_email(self, alert: Alert):
        """Send email alert"""
        if not self.smtp_config:
            logger.warning("SMTP not configured")
            return
        
        msg = MIMEMultipart()
        msg['Subject'] = f"[{alert.severity.name}] {alert.title}"
        msg['From'] = self.smtp_config.get('from', 'alerts@beverly-knits.com')
        msg['To'] = ', '.join(self._get_recipients(alert.severity))
        
        # Create HTML body
        html_body = f"""
        <html>
            <body>
                <h2>{alert.title}</h2>
                <p><strong>Severity:</strong> {alert.severity.name}</p>
                <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p>{alert.message}</p>
                <hr>
                <h3>Details:</h3>
                <pre>{json.dumps(alert.data, indent=2)}</pre>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send email
        with smtplib.SMTP(self.smtp_config['host'], self.smtp_config.get('port', 587)) as server:
            if self.smtp_config.get('use_tls', True):
                server.starttls()
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
    
    def _send_webhook(self, alert: Alert):
        """Send webhook notification"""
        webhook_url = self.webhook_urls.get(
            alert.severity.name.lower(),
            self.webhook_urls.get('default')
        )
        
        if not webhook_url:
            logger.warning("No webhook URL configured")
            return
        
        payload = {
            'alert_id': alert.id,
            'title': alert.title,
            'message': alert.message,
            'severity': alert.severity.name,
            'category': alert.category,
            'data': alert.data,
            'timestamp': alert.created_at.isoformat()
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    
    def _send_slack(self, alert: Alert):
        """Send Slack notification"""
        webhook_url = self.slack_config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("Slack webhook not configured")
            return
        
        # Color based on severity
        color_map = {
            AlertSeverity.CRITICAL: '#FF0000',
            AlertSeverity.HIGH: '#FF8C00',
            AlertSeverity.MEDIUM: '#FFD700',
            AlertSeverity.LOW: '#90EE90',
            AlertSeverity.INFO: '#87CEEB'
        }
        
        payload = {
            'text': alert.title,
            'attachments': [{
                'color': color_map.get(alert.severity, '#808080'),
                'title': alert.title,
                'text': alert.message,
                'fields': [
                    {
                        'title': 'Severity',
                        'value': alert.severity.name,
                        'short': True
                    },
                    {
                        'title': 'Category',
                        'value': alert.category,
                        'short': True
                    }
                ],
                'footer': 'Beverly Knits ERP',
                'ts': int(alert.created_at.timestamp())
            }]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    
    def _send_teams(self, alert: Alert):
        """Send Microsoft Teams notification"""
        webhook_url = self.teams_config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("Teams webhook not configured")
            return
        
        # Color based on severity
        color_map = {
            AlertSeverity.CRITICAL: 'FF0000',
            AlertSeverity.HIGH: 'FF8C00',
            AlertSeverity.MEDIUM: 'FFD700',
            AlertSeverity.LOW: '90EE90',
            AlertSeverity.INFO: '87CEEB'
        }
        
        payload = {
            '@type': 'MessageCard',
            '@context': 'https://schema.org/extensions',
            'themeColor': color_map.get(alert.severity, '808080'),
            'title': alert.title,
            'text': alert.message,
            'sections': [{
                'facts': [
                    {'name': 'Severity', 'value': alert.severity.name},
                    {'name': 'Category', 'value': alert.category},
                    {'name': 'Time', 'value': alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
                ]
            }]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    
    def _send_discord(self, alert: Alert):
        """Send Discord notification"""
        webhook_url = self.webhook_urls.get('discord')
        
        if not webhook_url:
            logger.warning("Discord webhook not configured")
            return
        
        # Color based on severity (decimal)
        color_map = {
            AlertSeverity.CRITICAL: 16711680,  # Red
            AlertSeverity.HIGH: 16753920,      # Orange
            AlertSeverity.MEDIUM: 16766720,    # Gold
            AlertSeverity.LOW: 9498256,        # Light Green
            AlertSeverity.INFO: 8900331        # Light Blue
        }
        
        payload = {
            'embeds': [{
                'title': alert.title,
                'description': alert.message,
                'color': color_map.get(alert.severity, 8421504),
                'fields': [
                    {
                        'name': 'Severity',
                        'value': alert.severity.name,
                        'inline': True
                    },
                    {
                        'name': 'Category',
                        'value': alert.category,
                        'inline': True
                    }
                ],
                'timestamp': alert.created_at.isoformat()
            }]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    
    def _send_sms(self, alert: Alert):
        """Send SMS alert for critical issues"""
        if not self.sms_config:
            logger.warning("SMS not configured")
            return
        
        # Example using Twilio
        try:
            from twilio.rest import Client
            
            client = Client(
                self.sms_config['account_sid'],
                self.sms_config['auth_token']
            )
            
            # Truncate message for SMS
            sms_message = f"[{alert.severity.name}] {alert.message[:140]}"
            
            for recipient in self.sms_config.get('recipients', []):
                client.messages.create(
                    body=sms_message,
                    from_=self.sms_config['from_number'],
                    to=recipient
                )
        except ImportError:
            logger.warning("Twilio not installed for SMS alerts")
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
    
    def _get_recipients(self, severity: AlertSeverity) -> List[str]:
        """Get email recipients based on severity"""
        recipients = self.smtp_config.get('recipients', {})
        
        if severity == AlertSeverity.CRITICAL:
            return recipients.get('critical', []) + recipients.get('all', [])
        elif severity == AlertSeverity.HIGH:
            return recipients.get('high', []) + recipients.get('all', [])
        elif severity == AlertSeverity.MEDIUM:
            return recipients.get('medium', []) + recipients.get('all', [])
        else:
            return recipients.get('all', [])
    
    def _severity_to_color(self, severity: AlertSeverity) -> str:
        """Convert severity to color code"""
        color_map = {
            AlertSeverity.CRITICAL: 'danger',
            AlertSeverity.HIGH: 'warning',
            AlertSeverity.MEDIUM: 'attention',
            AlertSeverity.LOW: 'good',
            AlertSeverity.INFO: 'info'
        }
        return color_map.get(severity, 'default')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return {
            'delivery_stats': self.delivery_stats,
            'total_rules': len(self.rules),
            'recent_alerts': len(self.alert_history),
            'queue_size': self.alert_queue.qsize()
        }
    
    def shutdown(self):
        """Shutdown alert manager"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        logger.info("Alert manager shutdown complete")