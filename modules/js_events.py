"""
JavaScript event injection module for browser event detection.
Handles Page Visibility API and window blur/focus events.
"""

import streamlit.components.v1 as components
from typing import Optional


def inject_browser_event_listener(component_key: str = "browser_events") -> str:
    """
    Inject JavaScript code to listen for browser events.
    Uses postMessage to communicate with Streamlit backend.
    
    Args:
        component_key: Unique key for the component
        
    Returns:
        HTML string with JavaScript event listeners
    """
    js_code = """
    <script>
    (function() {
        // Check if already initialized
        if (window.proctoringEventsInitialized) {
            return;
        }
        window.proctoringEventsInitialized = true;
        
        // Function to send event to Streamlit
        function sendEvent(eventType, details) {
            if (window.parent && window.parent.postMessage) {
                window.parent.postMessage({
                    type: 'proctoring_event',
                    eventType: eventType,
                    details: details || {},
                    timestamp: new Date().toISOString()
                }, '*');
            }
        }
        
        // Page Visibility API - detect tab switching
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                sendEvent('tab_switch', {
                    reason: 'tab_hidden',
                    timestamp: new Date().toISOString()
                });
            } else {
                sendEvent('tab_focus', {
                    reason: 'tab_visible',
                    timestamp: new Date().toISOString()
                });
            }
        });
        
        // Window blur - detect app/window switching
        window.addEventListener('blur', function() {
            sendEvent('window_blur', {
                reason: 'window_lost_focus',
                timestamp: new Date().toISOString()
            });
        });
        
        // Window focus - detect window regaining focus
        window.addEventListener('focus', function() {
            sendEvent('window_focus', {
                reason: 'window_gained_focus',
                timestamp: new Date().toISOString()
            });
        });
        
        // Also listen for beforeunload to detect closing
        window.addEventListener('beforeunload', function() {
            sendEvent('window_close', {
                reason: 'window_closing',
                timestamp: new Date().toISOString()
            });
        });
        
        // Send initialization confirmation
        sendEvent('initialized', {
            reason: 'event_listeners_ready',
            timestamp: new Date().toISOString()
        });
    })();
    </script>
    """
    
    return js_code


def create_browser_event_component(session_id: Optional[int] = None):
    """
    Create a Streamlit component that injects browser event listeners.
    Uses a polling mechanism to check for events.
    
    Args:
        session_id: Current session ID for logging events
    """
    html_content = f"""
    <div id="proctoring-events" style="display: none;"></div>
    <script>
    (function() {{
        // Check if already initialized
        if (window.proctoringEventsInitialized) {{
            return;
        }}
        window.proctoringEventsInitialized = true;
        window.proctoringEventQueue = [];
        
        // Function to store event (will be polled by Python)
        function storeEvent(eventType, details) {{
            const event = {{
                type: 'proctoring_event',
                eventType: eventType,
                details: details || {{}},
                timestamp: new Date().toISOString()
            }};
            window.proctoringEventQueue.push(event);
            // Keep only last 50 events
            if (window.proctoringEventQueue.length > 50) {{
                window.proctoringEventQueue.shift();
            }}
        }}
        
        // Page Visibility API - detect tab switching
        document.addEventListener('visibilitychange', function() {{
            if (document.hidden) {{
                storeEvent('tab_switch', {{
                    reason: 'tab_hidden',
                    timestamp: new Date().toISOString()
                }});
            }} else {{
                storeEvent('tab_focus', {{
                    reason: 'tab_visible',
                    timestamp: new Date().toISOString()
                }});
            }}
        }});
        
        // Window blur - detect app/window switching
        window.addEventListener('blur', function() {{
            storeEvent('window_blur', {{
                reason: 'window_lost_focus',
                timestamp: new Date().toISOString()
            }});
        }});
        
        // Window focus - detect window regaining focus
        window.addEventListener('focus', function() {{
            storeEvent('window_focus', {{
                reason: 'window_gained_focus',
                timestamp: new Date().toISOString()
            }});
        }});
        
        // Send initialization confirmation
        storeEvent('initialized', {{
            reason: 'event_listeners_ready',
            timestamp: new Date().toISOString()
        }});
    }})();
    </script>
    """
    
    # Create component (key parameter removed - not supported in this Streamlit version)
    components.html(
        html_content,
        height=0
    )


def get_browser_events_reader():
    """
    Create a component that reads browser events from JavaScript queue.
    Note: This is a simplified approach for the prototype.
    In production, you'd use a proper bidirectional component.
    
    Returns:
        HTML component that exposes event queue
    """
    html_content = """
    <script>
    // Expose event queue for reading (simplified approach)
    window.getProctoringEvents = function() {
        const events = window.proctoringEventQueue || [];
        window.proctoringEventQueue = []; // Clear after reading
        return JSON.stringify(events);
    };
    </script>
    <div id="event-reader" style="display: none;"></div>
    """
    return html_content
