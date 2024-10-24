from django.contrib import admin
from .models import (
    OriginalPlatform,
    FreelancingAccount,
    File,
    Assignment,
    Chat,
    Attachment,
    Message,
    AssignmentFile,
    AssignmentSubmission,
    RevisionFile,
    Revision,
    SearchTagPairs
)


@admin.register(OriginalPlatform)
class OriginalPlatformAdmin(admin.ModelAdmin):
    list_display = ('platform_name', 'platform_domain_name', 'platform_home_page_url', 'date_created')
    search_fields = ('platform_name', 'platform_domain_name')


@admin.register(FreelancingAccount)
class FreelancingAccountAdmin(admin.ModelAdmin):
    list_display = ('username', 'first_name', 'last_name', 'account_gmail', 'original_platform', 'date_created')
    search_fields = ('username', 'account_gmail', 'first_name', 'last_name')
    list_filter = ('original_platform',)


@admin.register(File)
class FileAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'origin', 'timestamp')
    list_filter = ('origin',)
    search_fields = ('file',)


class AssignmentFileInline(admin.TabularInline):
    model = AssignmentFile
    extra = 1
    filter_horizontal = ('files',)


class AssignmentSubmissionInline(admin.TabularInline):
    model = AssignmentSubmission
    extra = 1
    filter_horizontal = ('files',)


class RevisionInline(admin.TabularInline):
    model = Revision
    extra = 1
    filter_horizontal = ('revision_files',)


@admin.register(Assignment)
class AssignmentAdmin(admin.ModelAdmin):
    list_display = ('subject', 'agent', 'original_platform', 'original_account', 'timestamp', 'completion_deadline', 'completed', 'has_revisions')
    list_filter = ('completed', 'has_revisions', 'has_deposit_been_paid', 'original_platform')
    search_fields = ('subject', 'description')
    date_hierarchy = 'timestamp'
    inlines = [AssignmentFileInline, AssignmentSubmissionInline, RevisionInline]


class MessageInline(admin.TabularInline):
    model = Message
    extra = 1


@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ('assignment', 'timestamp')
    date_hierarchy = 'timestamp'
    inlines = [MessageInline]


class AttachmentInline(admin.TabularInline):
    model = Attachment
    extra = 1
    filter_horizontal = ('files',)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('chat', 'creator', 'timestamp')
    list_filter = ('creator',)
    search_fields = ('body',)
    date_hierarchy = 'timestamp'
    inlines = [AttachmentInline]


@admin.register(Attachment)
class AttachmentAdmin(admin.ModelAdmin):
    list_display = ('message',)
    filter_horizontal = ('files',)


@admin.register(AssignmentFile)
class AssignmentFileAdmin(admin.ModelAdmin):
    list_display = ('assignment',)
    filter_horizontal = ('files',)


@admin.register(AssignmentSubmission)
class AssignmentSubmissionAdmin(admin.ModelAdmin):
    list_display = ('assignment', 'version', 'date_completed', 'date_to_be_delivered', 'delivered')
    list_filter = ('delivered',)
    date_hierarchy = 'date_completed'
    filter_horizontal = ('files',)


@admin.register(RevisionFile)
class RevisionFileAdmin(admin.ModelAdmin):
    list_display = ('id',)
    filter_horizontal = ('files',)


@admin.register(Revision)
class RevisionAdmin(admin.ModelAdmin):
    list_display = ('assignment', 'assignment_submission', 'date_created', 'deadline', 'revision_price', 'needs_payment', 'has_deposit_been_paid')
    list_filter = ('needs_payment', 'has_deposit_been_paid')
    date_hierarchy = 'date_created'
    filter_horizontal = ('revision_files',)


@admin.register(SearchTagPairs)
class SearchTagPairsAdmin(admin.ModelAdmin):
    list_display = ('tag_one',)
    list_filter = ('tag_one','tag_two','tag_three')

# ip royal https://iproyal.com/?r=587935
