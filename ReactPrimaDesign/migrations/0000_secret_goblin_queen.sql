CREATE TABLE "chat_feedbacks" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"message_id" integer,
	"rating" integer,
	"feedback" text,
	"timestamp" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "chat_messages" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"role" text NOT NULL,
	"content" text NOT NULL,
	"timestamp" timestamp DEFAULT now() NOT NULL,
	"metadata" text
);
--> statement-breakpoint
CREATE TABLE "payment_methods" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"type" text NOT NULL,
	"name" text NOT NULL,
	"is_default" boolean DEFAULT false,
	"upi_id" text,
	"card_number" text,
	"expiry_date" text,
	"account_number" text,
	"ifsc_code" text,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "scam_reports" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"upi_id" text NOT NULL,
	"scam_type" text NOT NULL,
	"amount_lost" integer,
	"description" text,
	"timestamp" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "transactions" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"description" text DEFAULT 'Payment' NOT NULL,
	"upi_id" text NOT NULL,
	"amount" integer NOT NULL,
	"currency" text DEFAULT 'inr' NOT NULL,
	"transaction_type" text DEFAULT 'payment' NOT NULL,
	"status" text NOT NULL,
	"payment_method" text DEFAULT 'upi',
	"payment_intent_id" text,
	"timestamp" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "upi_risk_reports" (
	"id" serial PRIMARY KEY NOT NULL,
	"upi_id" text NOT NULL,
	"report_count" integer DEFAULT 0 NOT NULL,
	"first_report_date" timestamp DEFAULT now() NOT NULL,
	"risk_score" integer DEFAULT 0 NOT NULL,
	"status_verified" boolean DEFAULT false,
	"last_checked" timestamp
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" serial PRIMARY KEY NOT NULL,
	"username" text NOT NULL,
	"password" text NOT NULL,
	"phone_number" text NOT NULL,
	"name" text,
	"email" text,
	"address" text,
	"date_of_birth" text,
	"profile_completed" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"last_login" timestamp,
	"use_biometric" boolean DEFAULT false,
	"use_pin" boolean DEFAULT false,
	"pin" text,
	"device_id" text,
	CONSTRAINT "users_username_unique" UNIQUE("username"),
	CONSTRAINT "users_phone_number_unique" UNIQUE("phone_number")
);
--> statement-breakpoint
ALTER TABLE "chat_feedbacks" ADD CONSTRAINT "chat_feedbacks_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "chat_feedbacks" ADD CONSTRAINT "chat_feedbacks_message_id_chat_messages_id_fk" FOREIGN KEY ("message_id") REFERENCES "public"."chat_messages"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "chat_messages" ADD CONSTRAINT "chat_messages_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "payment_methods" ADD CONSTRAINT "payment_methods_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "scam_reports" ADD CONSTRAINT "scam_reports_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "transactions" ADD CONSTRAINT "transactions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;