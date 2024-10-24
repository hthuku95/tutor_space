from django.urls import path
from .views import (
    OriginalPlatformListView,
    OriginalPlatformDetailView,
    FreelancingAccountListView,
    FreelancingAccountDetailView,
    FileListView,
    FileDetailView,
    AssignmentListView,
    AssignmentDetailView,
    ChatListView,
    ChatDetailView,
    AttachmentListView,
    AttachmentDetailView,
    MessageListView,
    MessageDetailView,
    AssignmentFileListView,
    AssignmentFileDetailView,
    AssignmentSubmissionListView,
    AssignmentSubmissionDetailView,
    RevisionFileListView,
    RevisionFileDetailView,
    RevisionListView,
    RevisionDetailView,
    SearchTagPairsListView,
    TriggerBiddingView,
)

urlpatterns = [
    path('original-platforms/', OriginalPlatformListView.as_view(), name='original-platform-list'),
    path('original-platforms/<int:pk>/', OriginalPlatformDetailView.as_view(), name='original-platform-detail'),

    path('search-tag-pairs/', SearchTagPairsListView.as_view(), name='search-tag-pairs'),

    path('trigger-bidding/', TriggerBiddingView.as_view(), name='trigger-bidding'),

    path('freelancing-accounts/', FreelancingAccountListView.as_view(), name='freelancing-account-list'),
    path('freelancing-accounts/<int:pk>/', FreelancingAccountDetailView.as_view(), name='freelancing-account-detail'),

    path('files/', FileListView.as_view(), name='file-list'),
    path('files/<int:pk>/', FileDetailView.as_view(), name='file-detail'),

    path('assignments/', AssignmentListView.as_view(), name='assignment-list'),
    path('assignments/<int:pk>/', AssignmentDetailView.as_view(), name='assignment-detail'),

    path('chats/', ChatListView.as_view(), name='chat-list'),
    path('chats/<int:pk>/', ChatDetailView.as_view(), name='chat-detail'),

    path('attachments/', AttachmentListView.as_view(), name='attachment-list'),
    path('attachments/<int:pk>/', AttachmentDetailView.as_view(), name='attachment-detail'),

    path('messages/', MessageListView.as_view(), name='message-list'),
    path('messages/<int:pk>/', MessageDetailView.as_view(), name='message-detail'),

    path('assignment-files/', AssignmentFileListView.as_view(), name='assignment-file-list'),
    path('assignment-files/<int:pk>/', AssignmentFileDetailView.as_view(), name='assignment-file-detail'),

    path('assignment-submissions/', AssignmentSubmissionListView.as_view(), name='assignment-submission-list'),
    path('assignment-submissions/<int:pk>/', AssignmentSubmissionDetailView.as_view(), name='assignment-submission-detail'),

    path('revision-files/', RevisionFileListView.as_view(), name='revision-file-list'),
    path('revision-files/<int:pk>/', RevisionFileDetailView.as_view(), name='revision-file-detail'),

    path('revisions/', RevisionListView.as_view(), name='revision-list'),
    path('revisions/<int:pk>/', RevisionDetailView.as_view(), name='revision-detail'),
]
