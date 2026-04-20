import os.path
import json
import os
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from mcp.server.fastmcp import FastMCP

# If modifying these scopes, delete the file token_addon.json.
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify"
]

mcp = FastMCP("Gmail Addon")

def get_gmail_service():
    creds = None
    # We use the global credentials path managed by setup_google_auth.py
    global_creds_path = os.path.expanduser("~/.google_workspace_mcp/credentials/credentials.json")
    
    if os.path.exists(global_creds_path):
        creds = Credentials.from_authorized_user_file(global_creds_path, SCOPES)
    
    # If there are no (valid) credentials available, advise user to run setup script
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save the refreshed credentials back to the global path
                with open(global_creds_path, "w") as token:
                    token.write(creds.to_json())
            except Exception:
                raise Exception("Google credentials expired. Please run 'python setup_google_auth.py' in your terminal.")
        else:
            raise Exception("Authentication required. Please run 'python setup_google_auth.py' in your terminal to create credentials.")

    return build("gmail", "v1", credentials=creds)

@mcp.tool()
def get_total_unread_emails() -> str:
    """Get the absolute total count of unread emails in the user's inbox."""
    try:
        service = get_gmail_service()
        # Get label details for 'INBOX' or 'UNREAD'
        # Actually 'INBOX' typically has the unread count we want
        result = service.users().labels().get(userId="me", id="INBOX").execute()
        unread_count = result.get("messagesUnread", 0)
        total_count = result.get("messagesTotal", 0)
        
        return f"Total Unread Emails: {unread_count} (out of {total_count} total messages in Inbox)"
    except HttpError as error:
        return f"An error occurred: {error}"
    except Exception as e:
        return f"Failed: {str(e)}"

@mcp.tool()
def list_unread_message_ids(query: str = "is:unread", limit: int = 500) -> str:
    """Get a machine-readable JSON list of message IDs matching a query.
    
    Use this when you need to perform batch actions (mark as read, delete) 
    on multiple messages.
    """
    try:
        service = get_gmail_service()
        results = service.users().messages().list(userId="me", q=query, maxResults=limit).execute()
        messages = results.get("messages", [])
        ids = [m["id"] for m in messages]
        return json.dumps(ids)
    except HttpError as error:
        return f"An error occurred: {error}"
    except Exception as e:
        return f"Failed: {str(e)}"

@mcp.tool()
def batch_apply_labels_to_all(query: str = "is:unread", add_label_ids: list[str] = None, remove_label_ids: list[str] = None) -> str:
    """Apply or remove labels from ALL messages matching a query (handles pagination).
    
    Use this for bulk actions on the entire inbox or large sets of messages (1000+).
    """
    try:
        service = get_gmail_service()
        all_ids = []
        next_page = None
        
        # 1. Collect ALL IDs
        while True:
            results = service.users().messages().list(userId="me", q=query, pageToken=next_page).execute()
            messages = results.get("messages", [])
            all_ids.extend([m["id"] for m in messages])
            next_page = results.get("nextPageToken")
            if not next_page:
                break
        
        if not all_ids:
            return "No messages found matching the query."
            
        # 2. Batch modify in chunks of 1000
        count = 0
        for i in range(0, len(all_ids), 1000):
            batch = all_ids[i:i+1000]
            body = {"ids": batch}
            if add_label_ids: body["addLabelIds"] = add_label_ids
            if remove_label_ids: body["removeLabelIds"] = remove_label_ids
            
            service.users().messages().batchModify(userId="me", body=body).execute()
            count += len(batch)
            
        return f"Successfully processed {count} messages matching query '{query}'."
    except HttpError as error:
        return f"An error occurred: {error}"
    except Exception as e:
        return f"Failed: {str(e)}"

@mcp.tool()
def get_label_metrics(labels: list[str] = None) -> str:
    """Get total and unread message counts for specific labels.
    
    Args:
        labels: List of label IDs (e.g., ['INBOX', 'SPAM', 'TRASH', 'SENT']). Defaults to common labels.
    """
    if not labels:
        labels = ["INBOX", "SPAM", "TRASH", "SENT"]
    
    try:
        service = get_gmail_service()
        metrics = []
        for label_id in labels:
            try:
                result = service.users().labels().get(userId="me", id=label_id).execute()
                metrics.append({
                    "label": label_id,
                    "unread": result.get("messagesUnread", 0),
                    "total": result.get("messagesTotal", 0)
                })
            except HttpError:
                metrics.append({"label": label_id, "error": "Label not found or inaccessible"})
        
        return json.dumps(metrics, indent=2)
    except HttpError as error:
        return f"An error occurred: {error}"
    except Exception as e:
        return f"Failed: {str(e)}"

if __name__ == "__main__":
    mcp.run()
