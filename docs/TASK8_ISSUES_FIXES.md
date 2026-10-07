# Task 8 - Issues and Fixes Documentation

## 1. Overview

During Task 8 implementation and end-to-end testing, several issues were identified and fixed.

This document records the main issues, their causes, and the implemented fixes.

---

## 2. Issue: SQLite Row Data Handling

### Problem

The database functions return SQLite Row objects.

The Post-Interaction Summary module initially expected normal dictionary-style data.

This caused problems when accessing conversation and message information.

### Fix

A helper function was added to convert SQLite Row objects into normal dictionaries before processing the data.

### Result

Conversation and message data can now be processed correctly by the Post-Interaction Summary Agent.

**Status: Fixed**

---

## 3. Issue: Intent and Sentiment LLM Authentication

### Problem

The external Hugging Face/LLM service returned an authentication error when a valid API token was not available.

Example:

```text
401 Client Error: Unauthorized